import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from PIL import Image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

    # 向前传播函数
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 将剩余元素展平到一维
        x = self.fc_layers(x)
        return x


class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # 获取数据集的所有名称，返回名称列表

    def __len__(self):
        return sum([len(files) for _, _, files in os.walk(self.root_dir)])

    def __getitem__(self, idx):
        label = None
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)  # 获得子目录路径
            if idx < len(os.listdir(class_dir)):
                img_name = os.listdir(class_dir)[idx]
                img_path = os.path.join(class_dir, img_name)
                image = Image.open(img_path).convert('L')  # 转换成灰度图像
                label = i
                break
            idx -= len(os.listdir(class_dir))
        if self.transform:
            image = self.transform(image)
        return image, label


# 图像预处理操作
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 8  # 训练批次
num_epochs = 100  # 训练次数
learning_rate = 0.001  # 学习率

# 加载数据
train_dir = './data/train/'
train_dataset = Dataset(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dir = './data/val/'
val_dataset = Dataset(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

# 存储训练过程中的损失值
train_losses = []

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        train_losses.append(loss.item())  # 存储损失值

        if True:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model, 'model.pth')

# 绘制损失曲线
plt.plot(train_losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# 对测试集进行预测
model.eval()  # 将模型切换为评估模式
correct = 0
total = 0
predictions = []
true_labels = []

with torch.no_grad():  # 上下文管理器,关闭自动求梯度
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)  # 计算测试集样本总数量
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# 构建混淆矩阵
confusion = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='.20g', cmap=plt.cm.Blues)
plt.xlabel("Predicted label")
plt.ylabel("True Label")
plt.title("Testing Accuracy : {:.2f}%".format(accuracy))
plt.xticks(np.arange(2) + 0.5, val_dataset.classes)
plt.yticks(np.arange(2) + 0.5, val_dataset.classes)
plt.show()
