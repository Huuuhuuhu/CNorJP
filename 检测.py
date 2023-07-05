import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def predict_image(image_path):
    # 加载模型
    model = torch.load('model.pth')
    model.eval()

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 打开图像并进行预处理
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    # 使用模型进行预测
    output = model(image)
    _, predicted = torch.max(output, 1)
    prediction = predicted.item()  # 将模型的预测结果转换为 Python 中的整数类型
    if predicted == 0:
        label = '中国人'
    else:
        label = '日本人'
    result_label.config(text=f"鉴定为：{label}")


def detect_realtime():
    # 加载模型
    model = torch.load('model.pth')
    model.eval()

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 打开相机
    cap = cv2.VideoCapture(0)

    while True:
        # 读取相机帧
        ret, frame = cap.read()

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 将图像转换为 PIL Image
        image = Image.fromarray(gray)

        # 进行预处理
        image = transform(image).unsqueeze(0)

        # 使用模型进行预测
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

        if predicted == 0:
            label = '中国人'
        else:
            label = '日本人'

        # 在界面上显示预测结果
        result_label.config(text=f"鉴定为：{label}")

        # 显示实时图像
        cv2.imshow('Real-time Detection', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放相机资源
    cap.release()
    cv2.destroyAllWindows()


# 创建界面
root = tk.Tk()
root.title("国籍鉴定器")
root.geometry("400x350")

# 创建按钮和标签
button_image = tk.Button(root, text="选择图像", command=lambda: predict_image(filedialog.askopenfilename()), width=20, height=2)
button_image.pack(pady=10)

button_realtime = tk.Button(root, text="实时检测", command=detect_realtime, width=20, height=2)
button_realtime.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

# 运行界面主循环
root.mainloop()
