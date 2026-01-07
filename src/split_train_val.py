import os
import random
import shutil

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_SPLIT = 0.15  # 15%

os.makedirs(VAL_DIR, exist_ok=True)

for cls in os.listdir(TRAIN_DIR):
    cls_path = os.path.join(TRAIN_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    val_count = int(len(images) * VAL_SPLIT)
    val_images = images[:val_count]

    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    for img in val_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(VAL_DIR, cls, img)
        shutil.move(src, dst)

print("Train/Validation split completed.")
