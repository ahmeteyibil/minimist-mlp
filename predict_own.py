"""
Kendi el yazısı görüntülerini MNIST formatına getirip eğitilmiş MLP'ye verir.
Kullanım: python predict_own.py
"""

import os, numpy as np, torch, torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def preprocess(path, invert_if_white_bg=True):
    """Görüntüyü MNIST formatına getir: beyaz rakam, siyah zemin, 28x28, normalize."""
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.uint8)

    # Kâğıt beyazsa (ortalama > 127) renkleri ters çevir → MNIST gibi
    if invert_if_white_bg and arr.mean() > 127:
        arr = 255 - arr

    # Rakamın bounding box'ını bul, kırp
    thresh = arr > 30
    if thresh.any():
        ys, xs = np.where(thresh)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # En-boy oranını koru, en büyük kenarı 20 px yap (MNIST standardı)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    digit = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)

    # 28x28 tuvale merkezle
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - nw) // 2, (28 - nh) // 2))
    arr28 = np.array(canvas).astype(np.float32) / 255.0

    # Normalize (MNIST istatistikleri)
    arr28 = (arr28 - 0.1307) / 0.3081
    return arr28

def predict(model, arr28):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(arr28).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs

# ---------------------------
# Modeli yükle
# ---------------------------
model = MLP(hidden=(512, 256, 128)).to(DEVICE)
model.load_state_dict(torch.load("outputs/mlp_model.pth", map_location=DEVICE))
print("Model yüklendi.")

# ---------------------------
# Klasördeki tüm görüntüleri tahmin et
# ---------------------------
folder = "hand-writing"
files = sorted([f for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])

if not files:
    print("UYARI: hand-writing klasörüne görüntü ekleyin.")
    raise SystemExit

fig, axes = plt.subplots(1, len(files), figsize=(3 * len(files), 3.5))
if len(files) == 1:
    axes = [axes]

for ax, fname in zip(axes, files):
    path = os.path.join(folder, fname)
    arr = preprocess(path)
    pred, probs = predict(model, arr)

    # Görüntüyü orijinal haliyle (normalize geri al) göster
    show = arr * 0.3081 + 0.1307
    ax.imshow(show, cmap="gray")
    ax.set_title(f"{fname}\nTahmin: {pred} ({probs[pred]*100:.1f}%)",
                 fontsize=10)
    ax.axis("off")
    print(f"{fname:20s} → Tahmin: {pred}  | Olasılık: {probs[pred]:.4f}  "
          f"| Tüm olasılıklar: {np.round(probs, 3)}")

plt.tight_layout()
plt.savefig("outputs/own_predictions.png", dpi=150)
print("Kaydedildi: outputs/own_predictions.png")