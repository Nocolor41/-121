"""
从已解压的 Oxford-IIIT Pet 数据集中提取100张猫狗图片
"""
import os
import shutil

SRC_DIR = "./data_cache/oxford-iiit-pet/images"
DST_DIR = "./datasets"
os.makedirs(DST_DIR, exist_ok=True)

# 前12个品种是猫
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx"
}

cat_count = 0
dog_count = 0
TARGET = 50  # 每类50张

for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    # 解析品种名：文件名格式为 "BreedName_number.jpg"
    # 品种名可能包含下划线，如 "British_Shorthair_1.jpg"
    # 从后往前找第一个数字前的部分
    parts = fname.rsplit('_', 1)
    if len(parts) != 2:
        continue
    breed = parts[0]

    if breed in CAT_BREEDS:
        if cat_count >= TARGET:
            continue
        dst = os.path.join(DST_DIR, f"cat.{cat_count+1}.jpg")
        cat_count += 1
    else:
        if dog_count >= TARGET:
            continue
        dst = os.path.join(DST_DIR, f"dog.{dog_count+1}.jpg")
        dog_count += 1

    shutil.copy2(os.path.join(SRC_DIR, fname), dst)

    if cat_count >= TARGET and dog_count >= TARGET:
        break

print(f"完成！猫: {cat_count}张, 狗: {dog_count}张, 共: {cat_count+dog_count}张")
print(f"目录: {os.path.abspath(DST_DIR)}")

# 验证
files = os.listdir(DST_DIR)
print(f"验证: cat文件 {len([f for f in files if f.startswith('cat')])} 个, "
      f"dog文件 {len([f for f in files if f.startswith('dog')])} 个")
