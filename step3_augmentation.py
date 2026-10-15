"""
第三步：加入数据增强（预训练 ResNet18 + 数据增强）
对比：无增强 vs 有增强
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD


def build_resnet18(num_classes=2, freeze=True):
    """构建预训练 ResNet18，替换分类头"""
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if freeze:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.fc.parameters():
            p.requires_grad = True
    return net


if __name__ == '__main__':
    device = 'cpu'

    # ---- 实验A：无数据增强（基线）----
    print("=" * 60)
    print("第三步-A：ResNet18 预训练（无数据增强，基线）")
    print("=" * 60)
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    net_a = build_resnet18()
    acc_a, p_a, r_a, f1_a, time_a = train_and_eval(
        net_a, train_loader, val_loader, epochs=5, lr=1e-3, device=device
    )

    # ---- 实验B：加入数据增强 ----
    print("\n" + "=" * 60)
    print("第三步-B：ResNet18 预训练 + 数据增强")
    print("=" * 60)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_loader_b, val_loader_b = get_loaders(
        "./datasets", batch_size=16, train_transform=train_transform
    )
    net_b = build_resnet18()
    acc_b, p_b, r_b, f1_b, time_b = train_and_eval(
        net_b, train_loader_b, val_loader_b, epochs=5, lr=1e-3, device=device
    )

    print(f"\n【第三步对比结果】")
    print(f"{'指标':<12} {'无增强':<12} {'有增强':<12} {'变化':<12}")
    print("-" * 48)
    print(f"{'准确率':<12} {acc_a:<12.4f} {acc_b:<12.4f} {acc_b-acc_a:+.4f}")
    print(f"{'精确率':<12} {p_a:<12.4f} {p_b:<12.4f} {p_b-p_a:+.4f}")
    print(f"{'召回率':<12} {r_a:<12.4f} {r_b:<12.4f} {r_b-r_a:+.4f}")
    print(f"{'F1':<12} {f1_a:<12.4f} {f1_b:<12.4f} {f1_b-f1_a:+.4f}")
    print(f"{'训练用时':<12} {time_a:<12.1f} {time_b:<12.1f} {time_b-time_a:+.1f}s")

    import json
    result = {
        "step": "step3_augmentation",
        "no_aug": {"accuracy": acc_a, "precision": p_a, "recall": r_a, "f1": f1_a, "train_time": time_a},
        "with_aug": {"accuracy": acc_b, "precision": p_b, "recall": r_b, "f1": f1_b, "train_time": time_b},
    }
    with open("result_step3.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n结果已保存到 result_step3.json")
