"""
第五步：模型保存与单张图片推理测试
- 训练时保存最佳模型（早停思想）
- 加载模型进行单张图片推理
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from common import get_loaders

CLASSES = ["cat 猫", "dog 狗"]


def train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3,
                         save_path="best_model.pth", device="cpu"):
    """训练并保存验证集最佳模型"""
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                correct += (net(X).argmax(1) == Y).sum().item()
                total += Y.size(0)
        acc = correct / total
        print(f"  Epoch [{epoch+1}/{epochs}] Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f"    ^ 最佳模型已保存到 {save_path}")

    print(f"  最佳准确率: {best_acc:.4f}")
    return best_acc


def load_model(weight_path="best_model.pth", device="cpu"):
    """加载训练好的 ResNet18 模型"""
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    net.eval()
    return net


def predict(image_path, net, device="cpu"):
    """单张图片推理"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item(), probs


if __name__ == '__main__':
    print("=" * 60)
    print("第五步：模型保存与单张图片推理测试")
    print("=" * 60)

    device = 'cpu'

    # ---- 5.1 训练并保存最佳模型 ----
    print("\n[5.1] 训练 ResNet18 并保存最佳模型...")
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True

    best_acc = train_with_best_save(
        net, train_loader, val_loader, epochs=5, lr=1e-3,
        save_path="best_model.pth", device=device
    )

    # ---- 5.2 单张图片推理测试 ----
    print("\n[5.2] 单张图片推理测试...")
    net = load_model("best_model.pth", device)

    # 测试几张图片
    test_images = []
    for f in sorted(os.listdir("./datasets")):
        if f.startswith("cat.") and int(f.split('.')[1]) <= 3:
            test_images.append(os.path.join("./datasets", f))
        if f.startswith("dog.") and int(f.split('.')[1]) <= 3:
            test_images.append(os.path.join("./datasets", f))

    results = []
    for img_path in sorted(test_images):
        label, conf, probs = predict(img_path, net, device)
        true_label = "cat 猫" if "cat" in img_path.lower() else "dog 狗"
        correct = "✓" if label.split()[0] == true_label.split()[0] else "✗"
        print(f"  {correct} {os.path.basename(img_path):<15} "
              f"真实: {true_label:<8} 预测: {label:<8} 置信度: {conf:.2%}")
        results.append({
            "image": os.path.basename(img_path),
            "true_label": true_label,
            "pred_label": label,
            "confidence": conf,
            "correct": correct == "✓",
        })

    # 统计
    correct_count = sum(r["correct"] for r in results)
    print(f"\n  测试样本: {len(results)} 张, 正确: {correct_count} 张, "
          f"准确率: {correct_count/len(results):.2%}")

    import json
    with open("result_step5.json", "w") as f:
        json.dump({"best_val_acc": best_acc, "inference_tests": results}, f, indent=2, ensure_ascii=False)
    print("\n结果已保存到 result_step5.json")
