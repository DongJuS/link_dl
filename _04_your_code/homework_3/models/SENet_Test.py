import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


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


# ==================== Test Functions ====================

def test_model(model, test_loader, device):
    """모델 테스트 및 상세 결과 출력"""
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    # FashionMNIST 클래스 이름
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    all_predictions = []
    all_targets = []
    
    print("\n" + "="*70)
    print("Testing Model on FashionMNIST Test Dataset")
    print("="*70)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 클래스별 정확도 계산
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            current_acc = 100. * correct / total
            pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
    
    # 전체 정확도
    overall_accuracy = 100. * correct / total
    
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    print("="*70)
    
    # 클래스별 정확도
    print("\nPer-Class Accuracy:")
    print("-" * 70)
    print(f"{'Class Name':<15} {'Accuracy':<12} {'Correct/Total':<20}")
    print("-" * 70)
    
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"{class_names[i]:<15} {class_acc:>6.2f}%      "
                  f"{class_correct[i]:>4}/{class_total[i]:<4}")
        else:
            print(f"{class_names[i]:<15} N/A")
    
    print("-" * 70)
    
    # 가장 잘 인식한 클래스와 가장 못한 클래스
    class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                        for i in range(10)]
    best_class_idx = class_accuracies.index(max(class_accuracies))
    worst_class_idx = class_accuracies.index(min(class_accuracies))
    
    print(f"\nBest Performance : {class_names[best_class_idx]:<15} ({class_accuracies[best_class_idx]:.2f}%)")
    print(f"Worst Performance: {class_names[worst_class_idx]:<15} ({class_accuracies[worst_class_idx]:.2f}%)")
    print("="*70 + "\n")
    
    return overall_accuracy, class_accuracies


def load_and_test():
    """학습된 모델을 로드하여 테스트"""
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # 테스트 데이터 변환 (Normalization만)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 테스트 데이터셋 로드
    print("\nLoading FashionMNIST test dataset...")
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 모델 초기화
    print("\nInitializing SENet model...")
    model = SENet(num_classes=10, reduction=16).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 학습된 모델 로드
    checkpoint_path = 'best_model.pth'
    print(f"\nLoading trained model from '{checkpoint_path}'...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 체크포인트 정보 출력
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
        if 'best_acc' in checkpoint:
            print(f"Best validation accuracy during training: {checkpoint['best_acc']:.2f}%")
        
        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
        
    except FileNotFoundError:
        print(f"Error: Model file '{checkpoint_path}' not found!")
        print("Please make sure the model file exists in the current directory.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 테스트 실행
    overall_acc, class_accs = test_model(model, test_loader, device)
    
    # 최종 요약
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Model: SENet with SE Blocks (reduction=16)")
    print(f"Dataset: FashionMNIST Test Set (10,000 samples)")
    print(f"Final Test Accuracy: {overall_acc:.2f}%")
    print(f"Average Per-Class Accuracy: {np.mean(class_accs):.2f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    load_and_test()