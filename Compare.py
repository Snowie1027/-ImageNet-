# 导入库
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
from collections.abc import Iterable  # 修正错误

# 定义数据加载和处理函数
def load_data(txt_file, root_dir):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    images = []
    labels = []
    for line in lines:
        parts = line.strip().split()
        img_path = os.path.join(root_dir, parts[0])
        label = int(parts[1]) - 1  # 假设标签从1开始，我们需要将其转换为从0开始
        images.append(img_path)
        labels.append(label)

    return images, labels

class CUB200Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_images, train_labels = load_data('./CUB_200_2011/train.txt', './CUB_200_2011/images')
val_images, val_labels = load_data('./CUB_200_2011/test.txt', './CUB_200_2011/images')

train_dataset = CUB200Dataset(train_images, train_labels, transform=transform)
val_dataset = CUB200Dataset(val_images, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_model(pretrained=True, num_classes=200):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

criterion = torch.nn.CrossEntropyLoss()

# 学习率设置
learning_rates = [
    (5e-3, 5e-2),
    (1e-3, 1e-2),
    (5e-4, 5e-3)
]

num_epochs = 20

# 训练和验证函数
def train_and_evaluate(train_loader, val_loader, lr_pretrained, lr_fc, num_epochs=20, pretrained=True):
    model = initialize_model(pretrained=pretrained)
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': lr_fc},
        {'params': [param for name, param in model.named_parameters() if "fc" not in name], 'lr': lr_pretrained}
    ], momentum=0.9)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 创建日志目录
    log_dir = f'runs/lr_pretrained_{lr_pretrained}_lr_fc_{lr_fc}_pretrained_{pretrained}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"Starting training with lr_pretrained={lr_pretrained}, lr_fc={lr_fc}, pretrained={pretrained}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_text('Learning Rates', f'lr_pretrained: {lr_pretrained}, lr_fc: {lr_fc}', epoch)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        scheduler.step()

    writer.close()


# 运行训练和验证
for lr_pretrained, lr_fc in learning_rates:
    train_and_evaluate(train_loader, val_loader, lr_pretrained, lr_fc, num_epochs, pretrained=True)
    train_and_evaluate(train_loader, val_loader, lr_pretrained, lr_fc, num_epochs, pretrained=False)
