"""
第二步：在手动模型基础上加载 ImageNet 预训练参数（迁移学习）
- 冻结卷积层，只训练最后的全连接层
"""
import torch
import torch.nn as nn
from torchvision import models
from common import get_loaders, train_and_eval


class AlexNetManual(nn.Module):
    """与 torchvision AlexNet 结构一致，方便加载预训练权重"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    print("=" * 60)
    print("第二步：AlexNet + ImageNet 预训练（冻结卷积层）")
    print("=" * 60)

    device = 'cpu'
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)

    # 1. 下载官方预训练权重，加载到1000类结构
    print("正在下载 ImageNet 预训练权重...")
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)
    net.load_state_dict(official.state_dict())

    # 2. 把分类头换成2类
    net.classifier[6] = nn.Linear(4096, 2)

    # 3. 冻结卷积层，只训练 classifier
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {total:,} | 可训练参数量: {trainable:,}")

    acc, p, r, f1, train_time = train_and_eval(
        net, train_loader, val_loader, epochs=5, lr=1e-3, device=device
    )

    print(f"\n【第二步结果】")
    print(f"  准确率: {acc:.4f}")
    print(f"  精确率: {p:.4f}")
    print(f"  召回率: {r:.4f}")
    print(f"  F1:     {f1:.4f}")
    print(f"  训练用时: {train_time:.1f}s")

    import json
    result = {
        "step": "step2_transfer",
        "model": "AlexNet (ImageNet pretrained, frozen backbone)",
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "train_time": train_time,
        "total_params": total,
        "trainable_params": trainable,
        "epochs": 5,
        "lr": 1e-3,
    }
    with open("result_step2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("结果已保存到 result_step2.json")
