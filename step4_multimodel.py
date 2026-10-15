"""
第四步：调用 torchvision.models 封装好的模型横向对比
resnet18 / mobilenet_v3_small / shufflenet_v2_x1_0
"""
import time
import torch
import torch.nn as nn
from torchvision import models
from common import get_loaders, train_and_eval


def build_model(name, num_classes=2, freeze=True):
    """统一接口：加载预训练模型并替换分类头"""
    constructors = {
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "shufflenet_v2_x1_0": lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    }
    net = constructors[name]()

    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()

    if freeze:
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net


def measure_inference_time(net, device="cpu", warmup=5, runs=20):
    """测量单张图片推理时间（ms）"""
    net.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = net(x)
        start = time.time()
        for _ in range(runs):
            _ = net(x)
        elapsed = (time.time() - start) / runs * 1000
    return elapsed


if __name__ == '__main__':
    print("=" * 60)
    print("第四步：多模型横向对比")
    print("=" * 60)

    device = 'cpu'
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)

    results = {}
    model_names = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]

    for name in model_names:
        print(f"\n----- 模型: {name} -----")
        net = build_model(name)

        total_params = sum(p.numel() for p in net.parameters())
        print(f"  参数量: {total_params:,}")

        acc, p, r, f1, train_time = train_and_eval(
            net, train_loader, val_loader, epochs=5, lr=1e-3, device=device
        )

        infer_time = measure_inference_time(net, device)

        results[name] = {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "train_time": train_time,
            "params": total_params,
            "infer_time_ms": infer_time,
        }

        # 保存模型权重（第五步要用）
        torch.save(net.state_dict(), f"model_{name}.pth")
        print(f"  推理时间: {infer_time:.2f} ms/张")
        print(f"  模型已保存: model_{name}.pth")

    # 打印对比表格
    print("\n" + "=" * 80)
    print("【第四步 横向对比结果】")
    print("=" * 80)
    print(f"{'模型':<22} {'参数量':<12} {'准确率':<10} {'F1':<10} {'训练用时':<12} {'推理时间':<12}")
    print("-" * 80)
    for name in model_names:
        r = results[name]
        print(f"{name:<22} {r['params']:<12,} {r['accuracy']:<10.4f} {r['f1']:<10.4f} "
              f"{r['train_time']:<12.1f} {r['infer_time_ms']:<12.2f}")

    # 最佳模型保存为 best_model.pth（第五步用）
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n最佳模型: {best_name} (准确率 {results[best_name]['accuracy']:.4f})")
    import shutil
    shutil.copy(f"model_{best_name}.pth", "best_model.pth")
    print(f"已复制为 best_model.pth")

    import json
    with open("result_step4.json", "w") as f:
        json.dump(results, f, indent=2)
    print("结果已保存到 result_step4.json")
