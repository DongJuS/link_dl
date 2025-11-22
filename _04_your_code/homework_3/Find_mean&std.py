import os
from pathlib import Path
import torch
import wandb
from torch import nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
from utils import get_num_cpu_cores, is_linux, is_windows  # pyright: ignore
# BASE_PATH = str(Path(__file__).resolve())
# print(BASE_PATH)

# import sys

# # sys.path.append(BASE_PATH)
# sys.path.insert(0,BASE_PATH)

def get_fashion_mnist_data():
    data_path = os.getcwd() + 'a_fashion_mnist'
    f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())

    print("Num Train Samples : ", len(f_mnist_train))
    print("Sample Shape : ", f_mnist_train[0][0].shape)
    return f_mnist_train

def get_fashion_mnist_test_data():
    data_path = os.getcwd() + 'a_fashion_mnist'

    f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download = True)
    f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
    print("Num Train Samples : ", len(f_mnist_test))
    print("Sample Shape : ", f_mnist_test[0][0].shape)

    return f_mnist_test, f_mnist_test_images

def calculate_mean_std_v1():
    # 데이터 로드 (transform 없이)
    data_path = os.getcwd() + '/a_fashion_mnist'
    dataset = datasets.FashionMNIST(
        data_path, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()  # ToTensor만 적용
    )
    
    # DataLoader 생성
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),  # 전체 데이터를 한번에
        num_workers=0
    )
    
    # 전체 데이터 가져오기
    data = next(iter(loader))
    images = data[0]  # (N, C, H, W) 형태
    
    # Mean과 Std 계산
    mean = images.mean(dim=[0, 2, 3])  # 채널별 평균
    std = images.std(dim=[0, 2, 3])    # 채널별 표준편차
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    return mean, std

def calculate_mean_std_v2(dataset):
    """
    torch.stack을 사용한 mean과 std 계산
    """
    print("\n" + "#" * 50)
    print("Calculating Mean and Std")
    print("#" * 50)
    
    # 첫 번째 이미지 확인
    img_t, _ = dataset[0]
    print(f"type: {type(img_t)}")
    print(f"shape: {img_t.shape}")
    print(f"min, max: {img_t.min()}, {img_t.max()}")
    
    print("#" * 50)
    
    # 모든 이미지를 스택으로 쌓기
    imgs = torch.stack([img_t for img_t, _ in dataset], dim=3)
    print(f"imgs.shape: {imgs.shape}")
    
    # Mean과 Std 계산
    mean = imgs.view(1, -1).mean(dim=-1)
    std = imgs.view(1, -1).std(dim=-1)
    
    print(f"\nmean: {mean}")
    print(f"std: {std}")
    
    return mean, std
if __name__ == "__main__":
    config = {'' :0}
    wandb.init(mode="disabled", config=config)

    f_mnist_train= get_fashion_mnist_data()
    f_mnist_test_images,f_mnist_test = get_fashion_mnist_test_data()

    print(f_mnist_train, f_mnist_test_images, f_mnist_test)
    
    mean, std = calculate_mean_std_v1()
    mean, std = calculate_mean_std_v2(f_mnist_train)
    