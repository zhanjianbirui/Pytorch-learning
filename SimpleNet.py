import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ✅ 设备设置
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# -----------------------------------------------
# 1. 数据加载
# -----------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# -----------------------------------------------
# 2. 定义模型
# -----------------------------------------------


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleNet().to(device)  # ✅ 移到 MPS

# -----------------------------------------------
# 3. 损失函数 & 优化器
# -----------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------------------
# 4. 训练循环
# -----------------------------------------------
for epoch in range(10):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.view(-1, 784).to(device)  # ✅ 移到 MPS
        labels = labels.to(device)                # ✅ 移到 MPS

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10  |  Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------------------------
# 5. 评估
# -----------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 784).to(device)  # ✅ 移到 MPS
        labels = labels.to(device)                # ✅ 移到 MPS

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\n测试准确率: {100 * correct / total:.2f}%")

# -----------------------------------------------
# 6. 保存模型
# -----------------------------------------------
torch.save(model.state_dict(), "model_mps.pth")
print("模型已保存为 model_mps.pth")
