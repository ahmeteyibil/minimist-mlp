"""
MNIST üzerinde MLP tabanlı el yazısı rakam tanıma.
- Deney 1: 3 gizli katman, Adam, lr=1e-3
- Deney 2: 2 gizli katman, SGD+momentum, lr=0.01
CNN kullanılmamıştır. Hazır model yok, sıfırdan eğitim.
"""

import time, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ---------------------------
# Tekrarlanabilirlik
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------
# Veri seti: Train / Val / Test
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST ortalama/std
])

train_full = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds    = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

VAL_SIZE = 10000
train_size = len(train_full) - VAL_SIZE
train_ds, val_ds = random_split(
    train_full, [train_size, VAL_SIZE],
    generator=torch.Generator().manual_seed(SEED)
)

BATCH = 64
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256,   shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256,   shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ---------------------------
# MLP Modeli
# ---------------------------
class MLP(nn.Module):
    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        layers, in_dim = [], 28 * 28
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ---------------------------
# Eğitim & Değerlendirme fonksiyonları
# ---------------------------
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        tr_loss, tr_acc = run_loss / total, correct / total
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        print(f"Epoch {ep:2d}/{epochs} | train_loss={tr_loss:.4f} "
              f"acc={tr_acc:.4f} | val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}")
    elapsed = time.time() - t0
    return history, elapsed

def plot_history(history, title, fname):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_loss"], label="Train loss")
    ax[0].plot(ep, history["val_loss"],   label="Val loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[0].set_title(f"{title} – Loss"); ax[0].legend(); ax[0].grid(True)

    ax[1].plot(ep, history["train_acc"], label="Train acc")
    ax[1].plot(ep, history["val_acc"],   label="Val acc")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"{title} – Accuracy"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"Grafik kaydedildi: {fname}")

# ---------------------------
# DENEY 1 – Adam, 3 gizli katman
# ---------------------------
print("\n===== DENEY 1: Adam, 3 hidden layer =====")
model1 = MLP(hidden=(512, 256, 128)).to(DEVICE)
opt1   = optim.Adam(model1.parameters(), lr=1e-3)
crit   = nn.CrossEntropyLoss()

hist1, t1 = train_model(model1, train_loader, val_loader, opt1, crit, epochs=15)
test_loss1, test_acc1 = evaluate(model1, test_loader, crit)
print(f"[Deney 1] Test loss={test_loss1:.4f} | Test acc={test_acc1:.4f} | Süre={t1:.2f}s")

plot_history(hist1, "Deney 1 (Adam)", "outputs/exp1_curves.png")
torch.save(model1.state_dict(), "outputs/mlp_model.pth")

# ---------------------------
# DENEY 2 – SGD + momentum, 2 gizli katman
# ---------------------------
print("\n===== DENEY 2: SGD+momentum, 2 hidden layer =====")
model2 = MLP(hidden=(256, 128)).to(DEVICE)
opt2   = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
crit2  = nn.CrossEntropyLoss()

hist2, t2 = train_model(model2, train_loader, val_loader, opt2, crit2, epochs=15)
test_loss2, test_acc2 = evaluate(model2, test_loader, crit2)
print(f"[Deney 2] Test loss={test_loss2:.4f} | Test acc={test_acc2:.4f} | Süre={t2:.2f}s")

plot_history(hist2, "Deney 2 (SGD+momentum)", "outputs/exp2_curves.png")
torch.save(model2.state_dict(), "outputs/mlp_model_sgd.pth")

# ---------------------------
# Özet
# ---------------------------
summary = {
    "deney1": {
        "mimari": "784-512-256-128-10 (ReLU)",
        "optimizer": "Adam (lr=1e-3)",
        "epochs": 15, "batch": BATCH,
        "train_loss_son": hist1["train_loss"][-1],
        "val_loss_son":   hist1["val_loss"][-1],
        "train_acc_son":  hist1["train_acc"][-1],
        "val_acc_son":    hist1["val_acc"][-1],
        "test_loss": test_loss1, "test_acc": test_acc1,
        "egitim_suresi_sn": round(t1, 2),
    },
    "deney2": {
        "mimari": "784-256-128-10 (ReLU)",
        "optimizer": "SGD momentum=0.9 (lr=0.01)",
        "epochs": 15, "batch": BATCH,
        "train_loss_son": hist2["train_loss"][-1],
        "val_loss_son":   hist2["val_loss"][-1],
        "train_acc_son":  hist2["train_acc"][-1],
        "val_acc_son":    hist2["val_acc"][-1],
        "test_loss": test_loss2, "test_acc": test_acc2,
        "egitim_suresi_sn": round(t2, 2),
    }
}
with open("outputs/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nÖzet kaydedildi: outputs/summary.json")