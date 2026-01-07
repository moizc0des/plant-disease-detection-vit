import os

empty = []

for cls in os.listdir("data/val"):
    cls_path = os.path.join("data/val", cls)
    if os.path.isdir(cls_path) and len(os.listdir(cls_path)) == 0:
        empty.append(cls)

if not empty:
    print("✅ No empty validation folders")
else:
    print("❌ Empty validation folders:")
    for c in empty:
        print("-", c)
