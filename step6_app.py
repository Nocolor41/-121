"""
第六步：封装成可拍照分类的小程序（Gradio）
运行: python step6_app.py
访问: http://127.0.0.1:7860
"""
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms

CLASSES = {0: "猫 (Cat)", 1: "狗 (Dog)"}

# ---------- 1. 加载模型（启动时只做一次） ----------
net = models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, 2)
net.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True))
net.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------- 2. 定义预测函数 ----------
def predict(img: Image.Image):
    if img is None:
        return "请先上传或拍摄图片", {}
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    conf, pred = probs.max(0)
    detail = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
    result_text = f"识别结果：{CLASSES[int(pred)]}（置信度 {conf:.2%}）"
    return result_text, detail


# ---------- 3. 搭建界面 ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["webcam", "upload"], label="拍照或上传图片"),
    outputs=[
        gr.Textbox(label="识别结果"),
        gr.Label(label="各类别概率"),
    ],
    title="猫狗分类器 (Cat vs Dog Classifier)",
    description="基于 ResNet18 迁移学习训练。对准猫或狗拍照，或上传图片即可识别。",
    examples=[
        ["./datasets/cat.1.jpg"],
        ["./datasets/dog.1.jpg"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    print("=" * 60)
    print("第六步：Gradio 猫狗分类小程序")
    print("=" * 60)
    print("本机访问: http://127.0.0.1:7860")
    print("手机(同一WiFi)访问: http://<电脑IP>:7860")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=7860)
