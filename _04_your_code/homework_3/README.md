시도해볼만한 것들
증강
1. class별로 특히 신발은 제외하고 나머지에 대해서는 좌우 반전
2.  색깔을 임의로 넣어서 체킹하는것
3. pooling하는 부분.

근데 증강에 pooling이 들어가나 안들어가나 모르겠네.

[결과] a_fashion_mnist_data.py
Num Train Samples:  55000
Num Validation Samples:  5000
Sample Shape:  torch.Size([1, 28, 28])
Number of Data Loading Workers: 16

Num Test Samples:  10000
Sample Shape:  torch.Size([1, 28, 28])

다음 코드의 Mean과 Std값은 이번 [문제 1]에서 찾은 값으로 변경하기
f_mnist_transforms = nn.Sequential(
transforms.ConvertImageDtype(torch.float),
transforms.Normalize(mean=0.0, std=0.1), )
– 위 f_mnist_transforms 객체에 추가적인 transform	객체를 넣어 활용하는 것 허용
• 예를 들어 Image	Augmentation	기법 활용 가능

언급되는 교수님의 코딩 실력을 높이는 방법으로는
1. 로컬에서 GPU를 돌릴 수 있는 코드를 가지고, 학교 Koreatech GPU에서 실행하기
 - 이유는 코드를 자유자재로 돌릴 줄 알아야 하기 때문이다.

따라서 앞으로의 전략
1. 코드에서 사용되는 모든 데이터나 파일들의 경우에 같은 폴더에 전부 넣는다.
2. getcwd()라는 것을 활용해서 진행해보도록 하자.
