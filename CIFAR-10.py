import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),  # numpy 형태의 height ,width, channel 형태를
                                                        # torch 형태의 channel, height, width 형태로 바꾸어준다.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 정규화 진행 (input[channel] - mean[channel] / std[channel])

trainset = torchvision.datasets.CIFAR10(root='./data',  # CIFAR10를 저장할 위치
                                        train=True,  # train 데이터로 쓸것인지 여부
                                        download=True,  # 다운로드 여부
                                        transform=transform)  # 데이터 선처리

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,  # 배치 사이즈 크기 여부
                                          shuffle=True,  # 무작위로 이미지 선택
                                          num_workers=0)  # cpu 사용갯수

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize 즉, 이미지의 각 픽셀값을 0 ~ 1사이로 만들어 주기 위한 식
    npimg = img.numpy()  # tensor 형태라 다시 numpy형태로 변경
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # channel, height, width 순서를 다시 height, width, channel 순서로 변경

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))  # 이미지 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
plt.show()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # input channel, out channel, filter size
        self.pool = nn.MaxPool2d(2, 2)  # filter size 2, stride 2, padding 0
        self.conv2 = nn.Conv2d(6, 16, 5)  # input channel, out channel, filter size
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input features, out features
        self.fc2 = nn.Linear(120, 84)  # input features, out features
        self.fc3 = nn.Linear(84, 10)  # input features, out features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # activation function으로는 ReLu함수를 사용했다.
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # tensor 의 모양 변경
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = LeNet()

criterion = nn.CrossEntropyLoss()  # cross-entropy 사용
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 최적화 기법 SGD알고리즘 사용, learning rate, momentum 설정


m = []
n = []
for epoch in range(35):  # 2번의 epoch 실행
    total = 0.0
    running_loss = 0.0  # loss의 값을 0으로 초기화 해주는 변수
    for i, data in enumerate(trainloader, 0):  # train data를 하나씩 불러온다.
        inputs, labels = data  # 불러온 데이터를 input과 label에 넣는다.
        optimizer.zero_grad()  # gradient값을 초기화

        outputs = net(inputs)  # LeNet 모델에 input data 입력
        loss = criterion(outputs, labels)  # Loss function 구하기

        loss.backward()  # 역전파 진행
        optimizer.step()  # 역전파 진행된걸 가지고 weight를 업데이트 한다.

        running_loss += loss.item()

        if i % 2000 == 1999:  # 2000개가 학습될때마다 loss 측정
            total += running_loss / 2000
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    m.append(epoch + 1)
    n.append(total / 6)
plt.plot(m, n)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()  # testdata를 각 변수에 저장

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))  # 진짜 정답 출력
plt.show()

outputs = net(images)  # test image 예측하기위해 입력

_, predicted = torch.max(outputs, 1)  # softmax 원리를 통해 10개의 클래스 중 가장 높은 값 출력

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))  # 예측 값 출력
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # 가장 유력한 예측값을 뽑아낸다.
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 예측값과 실제값이 같으면 더한다.

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))  # 총 정확도를 총 데이터수로 나누어 확률로 출력한다.

class_correct = list(0. for i in range(10))  # 클래스 마다 예측할 수 입력할 리스트 생성
class_total = list(0. for i in range(10))  # 클래스 마다 카운터 수를 입력할 리스트 생성
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()  # 예측이 성공하면 차원을 축소하고 변수에 저장
        for i in range(4):  # 배치 사이즈만큼 반복
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
