import numpy as np
import torch
import timm
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"
MODEL_PATH = "model/vit_plantdoc.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, num_workers=0,
                                          pin_memory=(DEVICE.type == "cuda"))

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=28
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()


model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        preds = torch.argmax(out, 1).cpu()
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())


labels = np.unique(y_true)

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[test_ds.classes[i] for i in labels],
        zero_division=0
    )
)


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=test_ds.classes,
            yticklabels=test_ds.classes)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
