import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ✅ 设备
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# 10个类别名称
class_names = ['T恤', '裤子', '套头衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# ✅ 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ 数据集（只改这里！）
train_dataset = datasets.FashionMNIST(
    root='./data', train=True,  download=True, transform=transform)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ✅ 模型（和 SimpleNet 一样）


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # 加宽了网络
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.ASGD(model.parameters(), lr=0.001)

# ✅ 训练
for epoch in range(50):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/50  |  Loss: {total_loss/len(train_loader):.4f}")

# ✅ 评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\n测试准确率: {100 * correct / total:.2f}%")
# 全连接网络预期准确率约 88%
