"""
从 Oxford-IIIT Pet 数据集提取约100张猫狗图片，
按实验要求命名为 cat.N.jpg / dog.N.jpg，保存到 ./datasets
"""
import os
import shutil
import torchvision.datasets as datasets

DATA_DIR = "./datasets"
os.makedirs(DATA_DIR, exist_ok=True)

print("正在下载 Oxford-IIIT Pet 数据集（约800MB，首次需要几分钟）...")
dataset = datasets.OxfordIIITPet(
    root="./data_cache",
    split="trainval",
    target_types="category",
    download=True
)

# Oxford-IIIT Pet: 前12类是猫，后25类是狗
# 类别索引: 0-11 是猫, 12-36 是狗
cat_count = 0
dog_count = 0
TARGET_PER_CLASS = 50  # 每类50张，共100张

print(f"开始提取图片（猫{TARGET_PER_CLASS}张 + 狗{TARGET_PER_CLASS}张）...")

for idx in range(len(dataset)):
    img, label = dataset[idx]
    img_path = dataset._images[idx]

    if label <= 11:  # 猫
        if cat_count >= TARGET_PER_CLASS:
            continue
        dst = os.path.join(DATA_DIR, f"cat.{cat_count+1}.jpg")
        cat_count += 1
    else:  # 狗
        if dog_count >= TARGET_PER_CLASS:
            continue
        dst = os.path.join(DATA_DIR, f"dog.{dog_count+1}.jpg")
        dog_count += 1

    shutil.copy2(img_path, dst)

    if cat_count >= TARGET_PER_CLASS and dog_count >= TARGET_PER_CLASS:
        break

print(f"完成！共提取 {cat_count} 张猫 + {dog_count} 张狗 = {cat_count+dog_count} 张图片")
print(f"保存目录: {os.path.abspath(DATA_DIR)}")

# 验证
files = os.listdir(DATA_DIR)
cats = [f for f in files if f.startswith("cat")]
dogs = [f for f in files if f.startswith("dog")]
print(f"验证: cat文件 {len(cats)} 个, dog文件 {len(dogs)} 个")
