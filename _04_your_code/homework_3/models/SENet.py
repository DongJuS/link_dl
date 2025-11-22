import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import wandb
from tqdm import tqdm
import random

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==================== Data Augmentation ====================

class MixupTransform:
    """Mixup 데이터 증강"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_data, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_data.size(0)
        index = torch.randperm(batch_size).to(batch_data.device)
        
        mixed_data = lam * batch_data + (1 - lam) * batch_data[index]
        labels_a, labels_b = batch_labels, batch_labels[index]
        
        return mixed_data, labels_a, labels_b, lam


class CutMixTransform:
    """CutMix 데이터 증강"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_data, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_data.size(0)
        index = torch.randperm(batch_size).to(batch_data.device)
        
        # CutMix 영역 계산
        _, _, H, W = batch_data.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_data = batch_data.clone()
        mixed_data[:, :, bby1:bby2, bbx1:bbx2] = batch_data[index, :, bby1:bby2, bbx1:bbx2]
        
        # 실제 lambda 재계산
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        labels_a, labels_b = batch_labels, batch_labels[index]
        
        return mixed_data, labels_a, labels_b, lam


# ==================== SE Block ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==================== SENet Model ====================

class SENetBlock(nn.Module):
    """SENet의 기본 블록"""
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SENetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SENet(nn.Module):
    """FashionMNIST용 SENet"""
    def __init__(self, num_classes=10, reduction=16):
        super(SENet, self).__init__()
        
        self.in_channels = 32
        
        # 초기 Conv 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # SE 블록들
        self.layer1 = self._make_layer(32, 2, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(64, 2, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(128, 2, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(256, 2, stride=2, reduction=reduction)
        
        # Local Response Normalization
        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        
        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride, reduction):
        layers = []
        layers.append(SENetBlock(self.in_channels, out_channels, stride, reduction))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(SENetBlock(out_channels, out_channels, 1, reduction))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.lrn(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


# ==================== Early Stopping ====================

class EarlyStopping:
    """Early Stopping 구현"""
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


# ==================== Training Functions ====================

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix용 손실 함수"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, train_loader, criterion, optimizer, device, mixup, cutmix, use_mixing=True):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Mixing 적용 (50% 확률)
        if use_mixing and random.random() > 0.5:
            if random.random() > 0.5:
                # Mixup
                data, targets_a, targets_b, lam = mixup(data, target)
            else:
                # CutMix
                data, targets_a, targets_b, lam = cutmix(data, target)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / len(val_loader),
                'acc': 100. * correct / total
            })
    
    return running_loss / len(val_loader), 100. * correct / total


# ==================== Main Training ====================

def main():
    # 하이퍼파라미터
    config = {
        'batch_size': 128,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'mixup_alpha': 1.0,
        'cutmix_alpha': 1.0,
        'se_reduction': 16,
        'early_stopping_patience': 15,
    }
    
    # Wandb 초기화
    wandb.init(
        project='fashionmnist-senet',
        config=config,
        name='SENet-Mixup-CutMix'
    )
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터 변환
    # 학습 데이터: Padding + RandomCrop + Normalization
    train_transform = transforms.Compose([
        transforms.Pad(4, fill=0),  # 28x28 -> 36x36
        transforms.RandomCrop(28),  # 36x36 -> 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 검증/테스트 데이터: Normalization만
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 데이터셋 로드
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 초기화
    model = SENet(num_classes=10, reduction=config['se_reduction']).to(device)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Wandb로 모델 추적
    wandb.watch(model, log='all', log_freq=100)
    
    # 손실 함수 및 최적화
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Mixing 변환
    mixup = MixupTransform(alpha=config['mixup_alpha'])
    cutmix = CutMixTransform(alpha=config['cutmix_alpha'])
    
    # Early Stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        verbose=True
    )
    
    # 학습 루프
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        print(f'\n=== Epoch {epoch+1}/{config["epochs"]} ===')
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, mixup, cutmix
        )
        
        # 검증
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Wandb 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 최고 정확도 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
        
        # Early Stopping 체크
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping triggered!')
            model.load_state_dict(early_stopping.best_model)
            break
    
    # 최종 테스트
    print('\n=== Final Test ===')
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    wandb.log({
        'final_test_loss': test_loss,
        'final_test_acc': test_acc
    })
    
    wandb.finish()


if __name__ == '__main__':
    main()