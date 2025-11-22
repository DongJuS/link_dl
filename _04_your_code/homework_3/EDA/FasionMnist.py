import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터셋 다운로드 및 로드
# 전처리 없이 원본 이미지를 보기 위해 ToTensor만 적용합니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 2. 클래스 매핑 (숫자 Label -> 실제 옷 이름)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 3. 시각화 함수 정의
def visualize_dataset_grid(dataset, num_samples_per_class=10):
    fig = plt.figure(figsize=(15, 15))
    
    # 각 클래스별로 이미지를 수집할 딕셔너리
    class_samples = {i: [] for i in range(10)}
    
    # 데이터를 순회하며 클래스별로 샘플 수집 (속도를 위해 단순 순회)
    for img, label in dataset:
        if len(class_samples[label]) < num_samples_per_class:
            class_samples[label].append(img)
        
        # 모든 클래스가 꽉 차면 중단
        if all(len(samples) == num_samples_per_class for samples in class_samples.values()):
            break
    
    # 그리드에 그리기
    idx = 1
    for label_idx in range(10):
        for sample_idx in range(num_samples_per_class):
            ax = fig.add_subplot(10, num_samples_per_class, idx)
            
            # Tensor (C, H, W) -> Numpy (H, W, C) 로 변환 및 흑백 처리
            image = class_samples[label_idx][sample_idx].squeeze()
            
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            
            # 첫 번째 열에만 클래스 이름 붙이기
            if sample_idx == 0:
                ax.set_title(labels_map[label_idx], fontsize=12, fontweight='bold', loc='left')
            
            idx += 1
            
    plt.tight_layout()
    plt.show()

# 4. 실행
print("데이터 시각화를 시작합니다...")
visualize_dataset_grid(training_data)