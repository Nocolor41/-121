"""
主运行脚本：依次执行六步实验，收集所有结果
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD

DEVICE = 'cpu'
RESULTS = {}

def save_result(key, data):
    RESULTS[key] = data
    with open("all_results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)

# ============================================================
# 第一步：手动搭建 AlexNet，从零训练
# ============================================================
def step1():
    print("\n" + "=" * 70)
    print("第一步：手动搭建 AlexNet，从零训练（随机初始化）")
    print("=" * 70)

    class AlexNet(nn.Module):
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
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    net = AlexNet(num_classes=2)
    net.apply(init_weight)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params:,}")

    acc, p, r, f1, train_time = train_and_eval(
        net, train_loader, val_loader, epochs=10, lr=1e-4, device=DEVICE
    )
    result = {
        "model": "AlexNet (from scratch)",
        "accuracy": round(acc, 4), "precision": round(p, 4),
        "recall": round(r, 4), "f1": round(f1, 4),
        "train_time": round(train_time, 1), "params": total_params,
        "epochs": 10, "lr": 1e-4,
    }
    save_result("step1", result)
    print(f"结果: 准确率={acc:.4f}, F1={f1:.4f}, 用时={train_time:.1f}s")
    return result

# ============================================================
# 第二步：AlexNet + ImageNet 预训练（冻结卷积层）
# ============================================================
def step2():
    print("\n" + "=" * 70)
    print("第二步：AlexNet + ImageNet 预训练（冻结卷积层）")
    print("=" * 70)

    class AlexNetManual(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(), nn.Linear(256*6*6, 4096), nn.ReLU(inplace=True),
                nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    print("加载 ImageNet 预训练权重...")
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)
    net.load_state_dict(official.state_dict())
    net.classifier[6] = nn.Linear(4096, 2)

    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    total_params = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} | 可训练: {trainable:,}")

    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    acc, p, r, f1, train_time = train_and_eval(
        net, train_loader, val_loader, epochs=5, lr=1e-3, device=DEVICE
    )
    result = {
        "model": "AlexNet (pretrained, frozen)",
        "accuracy": round(acc, 4), "precision": round(p, 4),
        "recall": round(r, 4), "f1": round(f1, 4),
        "train_time": round(train_time, 1),
        "total_params": total_params, "trainable_params": trainable,
        "epochs": 5, "lr": 1e-3,
    }
    save_result("step2", result)
    print(f"结果: 准确率={acc:.4f}, F1={f1:.4f}, 用时={train_time:.1f}s")
    return result

# ============================================================
# 第三步：数据增强对比（ResNet18 预训练）
# ============================================================
def step3():
    print("\n" + "=" * 70)
    print("第三步：数据增强对比（ResNet18 预训练）")
    print("=" * 70)

    def build_resnet18():
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, 2)
        for p in net.parameters():
            p.requires_grad = False
        for p in net.fc.parameters():
            p.requires_grad = True
        return net

    # 3A: 无增强
    print("\n--- 3A: 无数据增强（基线）---")
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    net_a = build_resnet18()
    acc_a, p_a, r_a, f1_a, time_a = train_and_eval(
        net_a, train_loader, val_loader, epochs=5, lr=1e-3, device=DEVICE
    )

    # 3B: 有增强
    print("\n--- 3B: 加入数据增强 ---")
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
        net_b, train_loader_b, val_loader_b, epochs=5, lr=1e-3, device=DEVICE
    )

    result = {
        "no_aug": {"accuracy": round(acc_a,4), "precision": round(p_a,4),
                   "recall": round(r_a,4), "f1": round(f1_a,4), "train_time": round(time_a,1)},
        "with_aug": {"accuracy": round(acc_b,4), "precision": round(p_b,4),
                     "recall": round(r_b,4), "f1": round(f1_b,4), "train_time": round(time_b,1)},
    }
    save_result("step3", result)
    print(f"\n对比: 无增强 acc={acc_a:.4f} | 有增强 acc={acc_b:.4f} | 变化={acc_b-acc_a:+.4f}")
    return result

# ============================================================
# 第四步：多模型横向对比
# ============================================================
def step4():
    print("\n" + "=" * 70)
    print("第四步：多模型横向对比")
    print("=" * 70)

    def build_model(name):
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            net.fc = nn.Linear(net.fc.in_features, 2)
            head = net.fc.parameters()
        elif name == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            net.classifier[3] = nn.Linear(net.classifier[3].in_features, 2)
            head = net.classifier[3].parameters()
        elif name == "shufflenet_v2_x1_0":
            net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
            net.fc = nn.Linear(net.fc.in_features, 2)
            head = net.fc.parameters()
        for p in net.parameters():
            p.requires_grad = False
        for p in head:
            p.requires_grad = True
        return net

    def measure_inference(net):
        net.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            for _ in range(5):
                _ = net(x)
            start = time.time()
            for _ in range(20):
                _ = net(x)
            return (time.time() - start) / 20 * 1000

    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    results = {}
    for name in ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]:
        print(f"\n--- 模型: {name} ---")
        net = build_model(name)
        params = sum(p.numel() for p in net.parameters())
        acc, p, r, f1, train_time = train_and_eval(
            net, train_loader, val_loader, epochs=5, lr=1e-3, device=DEVICE
        )
        infer_time = measure_inference(net)
        torch.save(net.state_dict(), f"model_{name}.pth")
        results[name] = {
            "accuracy": round(acc,4), "precision": round(p,4),
            "recall": round(r,4), "f1": round(f1,4),
            "train_time": round(train_time,1), "params": params,
            "infer_time_ms": round(infer_time, 2),
        }
        print(f"  参数量={params:,}, 准确率={acc:.4f}, 推理={infer_time:.2f}ms")

    # 保存最佳模型
    best = max(results, key=lambda k: results[k]["accuracy"])
    import shutil
    shutil.copy(f"model_{best}.pth", "best_model.pth")
    print(f"\n最佳模型: {best}")

    save_result("step4", results)
    return results

# ============================================================
# 第五步：模型保存与单张图片推理
# ============================================================
def step5():
    print("\n" + "=" * 70)
    print("第五步：模型保存与单张图片推理测试")
    print("=" * 70)

    # 训练并保存最佳模型
    print("\n[5.1] 训练 ResNet18 并保存最佳模型...")
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True

    net.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    best_acc = 0.0
    for epoch in range(5):
        net.train()
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                correct += (net(X).argmax(1) == Y).sum().item()
                total += Y.size(0)
        acc = correct / total
        print(f"  Epoch [{epoch+1}/5] Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), "best_model.pth")
            print(f"    ^ 最佳模型已保存")
    print(f"  最佳准确率: {best_acc:.4f}")

    # 单张图片推理
    print("\n[5.2] 单张图片推理测试...")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True))
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    CLASSES = ["cat 猫", "dog 狗"]

    test_images = []
    for f in sorted(os.listdir("./datasets")):
        if f.startswith("cat.") and int(f.split('.')[1]) <= 3:
            test_images.append(f)
        if f.startswith("dog.") and int(f.split('.')[1]) <= 3:
            test_images.append(f)

    inference_results = []
    for fname in sorted(test_images):
        img = torch.jit if False else None  # placeholder
        from PIL import Image
        img = Image.open(os.path.join("./datasets", fname)).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)[0]
        pred = probs.argmax().item()
        true_label = "cat 猫" if "cat" in fname else "dog 狗"
        correct = CLASSES[pred].split()[0] == true_label.split()[0]
        print(f"  {'✓' if correct else '✗'} {fname:<15} 真实:{true_label:<8} "
              f"预测:{CLASSES[pred]:<8} 置信度:{probs[pred]:.2%}")
        inference_results.append({
            "image": fname, "true": true_label, "pred": CLASSES[pred],
            "confidence": round(probs[pred].item(), 4), "correct": correct,
        })

    result = {
        "best_val_acc": round(best_acc, 4),
        "inference_tests": inference_results,
        "inference_accuracy": round(sum(r["correct"] for r in inference_results) / len(inference_results), 4),
    }
    save_result("step5", result)
    return result

# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("猫狗图像分类实验 - 全流程运行")
    print(f"设备: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"数据集: ./datasets ({len(os.listdir('./datasets'))} 张图片)")

    total_start = time.time()

    step1()
    step2()
    step3()
    step4()
    step5()

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"全部实验完成！总用时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("结果已保存到 all_results.json")
    print("=" * 70)

    # 打印汇总
    print("\n【实验结果汇总】")
    print(f"第一步 从零训练:     准确率={RESULTS['step1']['accuracy']:.4f}")
    print(f"第二步 迁移学习:     准确率={RESULTS['step2']['accuracy']:.4f}")
    print(f"第三步 无增强:       准确率={RESULTS['step3']['no_aug']['accuracy']:.4f}")
    print(f"第三步 有增强:       准确率={RESULTS['step3']['with_aug']['accuracy']:.4f}")
    for name, r in RESULTS['step4'].items():
        print(f"第四步 {name:<20}: 准确率={r['accuracy']:.4f}")
    print(f"第五步 最佳模型:     准确率={RESULTS['step5']['best_val_acc']:.4f}")
