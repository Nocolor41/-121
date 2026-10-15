"""
第一步：手动搭建 AlexNet，从零训练（随机初始化权重）
"""
import torch
import torch.nn as nn
from common import get_loaders, train_and_eval


class AlexNet(nn.Module):
    """精简版 AlexNet（适配224x224输入，2分类输出）"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def init_weight(m):
    """Xavier 均匀初始化"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    print("=" * 60)
    print("第一步：手动搭建 AlexNet，从零训练")
    print("=" * 60)

    device = 'cpu'
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)

    net = AlexNet(num_classes=2)
    net.apply(init_weight)

    # 统计参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params:,}")

    acc, p, r, f1, train_time = train_and_eval(
        net, train_loader, val_loader, epochs=10, lr=1e-4, device=device
    )

    print(f"\n【第一步结果】")
    print(f"  准确率: {acc:.4f}")
    print(f"  精确率: {p:.4f}")
    print(f"  召回率: {r:.4f}")
    print(f"  F1:     {f1:.4f}")
    print(f"  训练用时: {train_time:.1f}s")

    # 保存结果供报告使用
    import json
    result = {
        "step": "step1_scratch",
        "model": "AlexNet (from scratch)",
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "train_time": train_time,
        "params": total_params,
        "epochs": 10,
        "lr": 1e-4,
    }
    with open("result_step1.json", "w") as f:
        json.dump(result, f, indent=2)
    print("结果已保存到 result_step1.json")
