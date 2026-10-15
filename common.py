"""
公共代码：数据加载 + 训练评估函数
每个实验步骤都会 import 本文件
"""
import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FileNameDataset(Dataset):
    """从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        label = 0 if 'cat' in img_name.lower() else 1
        if self.transform:
            image = self.transform(image)
        return image, label


def get_loaders(data_dir="./datasets", batch_size=16, train_transform=None, seed=42):
    """统一划分：80% 训练 / 20% 验证，固定随机种子保证各实验可比"""
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = FileNameDataset(image_dir=data_dir, transform=train_transform)
    val_set = FileNameDataset(image_dir=data_dir, transform=val_transform)

    g = torch.Generator().manual_seed(seed)
    total = len(train_set)
    train_size = int(0.8 * total)
    val_size = total - train_size
    idx_train, idx_val = random_split(range(total), [train_size, val_size], generator=g)

    train_loader = DataLoader(Subset(train_set, idx_train.indices),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_set, idx_val.indices),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu", verbose=True):
    """训练并评估，返回 (准确率, 精确率, 召回率, F1, 训练用时秒)"""
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        if verbose:
            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader.dataset):.4f}")

    train_time = time.time() - start_time

    # 评估
    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, Y in val_loader:
            preds = net(X.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(Y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                  average='macro', zero_division=0)
    if verbose:
        print(f"  准确率: {acc:.4f} | 精确率: {p:.4f} | 召回率: {r:.4f} | F1: {f1:.4f} | 用时: {train_time:.1f}s")
    return acc, p, r, f1, train_time
