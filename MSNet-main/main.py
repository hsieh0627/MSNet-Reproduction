from dataset import coast
from dataset import gulfport
from model import MSNet
import matplotlib.pyplot as plt
from torch.optim import Adam
import select_bands
import torch
import utils
import metric
import os
import numpy as np
from SeT import (
    TotalLoss,
    Mask,
    separation_training
)

# =========================
# Helper functions
# =========================
def normalize_img(img):
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img

def hsi_to_rgb(data):
    """
    將 HSI 轉成可顯示的 RGB 圖
    若 band 數 >= 3，取三個波段組成假 RGB
    """
    bands = data.shape[2]

    if bands >= 3:
        b1 = 0
        b2 = bands // 2
        b3 = bands - 1
        rgb = np.stack([data[:, :, b1], data[:, :, b2], data[:, :, b3]], axis=-1)
    else:
        # 若波段不足 3，重複堆成 3 channels
        gray = data[:, :, 0]
        rgb = np.stack([gray, gray, gray], axis=-1)

    rgb = normalize_img(rgb)
    return rgb

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

# =========================
# Settings
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lmda = 1e-3
num_bs = 64
num_layers = 3
lr = 1e-3
epochs = 150
output_iter = 5
max_iter = 10
data_norm = True
Net = MSNet
net_kwargs = dict()
net_kwargs['num_layers'] = num_layers

# =========================
# Load data
# =========================
# Load data
# =========================
# dataset = coast
dataset = coast   # 想跑哪個資料集就改這行

data, gt = dataset.get_data()
rows, cols, bands = data.shape
net_kwargs['shape'] = (rows, cols, num_bs)
print('Detecting on %s...' % dataset.name)

# =========================
# Preprocessing
# =========================
band_idx = select_bands.OPBS(data, num_bs)
data_bs = data[:, :, band_idx]
if data_norm:
    data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

# =========================
# Load model
# =========================
model = Net(**net_kwargs).to(device).float()

# Loss
loss = TotalLoss(lmda, device)

# Mask
mask = Mask((rows, cols), device)

# Optimizer
optimizer = Adam(model.parameters(), lr=lr)

# =========================
# Separation Training
# =========================
x_bs = torch.from_numpy(data_bs).to(device).float()
pr_dm, history = separation_training(
    x=x_bs,
    gt=gt,
    model=model,
    loss=loss,
    mask=mask,
    optimizer=optimizer,
    epochs=epochs,
    output_iter=output_iter,
    max_iter=max_iter,
    verbose=True
)

# 轉 numpy
pr_dm = to_numpy(pr_dm)
gt = to_numpy(gt)

# =========================
# Save the detection result
# =========================
result_path = os.path.join('results', model.name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

# =========================
# ROC curve -> PNG
# =========================
rx_dm = utils.rx(data)
rx_dm = to_numpy(rx_dm)

fpr, tpr, rx_auc = metric.roc_auc(rx_dm, gt)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label='RX: %.4f' % rx_auc)

fpr, tpr, pr_auc = metric.roc_auc(pr_dm, gt)
plt.plot(fpr, tpr, label='%s+SeT: %.4f' % (model.name, pr_auc),
         c='black', alpha=0.7)

plt.grid(alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('%s ROC Curve' % dataset.name)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(result_path, '%s_roc.png' % dataset.name),
    dpi=300
)
plt.close()

# =========================
# AUC history -> PNG
# =========================
iters = [(_ + 1) * epochs for _ in range(max_iter)]
plt.figure(figsize=(6, 5))
plt.xticks(iters)
plt.plot(iters, history, marker='o')
plt.scatter([output_iter * epochs], [history[output_iter - 1]],
            marker='o', edgecolors='black', facecolors='white', label='Stop',
            zorder=10)
plt.grid(alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('%s AUC History' % dataset.name)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(result_path, '%s_auc_history.png' % dataset.name),
    dpi=300
)
plt.close()

# =========================
# Save heatmap only -> PNG
# =========================
plt.figure(figsize=(6, 5))
plt.imshow(pr_dm, cmap='jet')
plt.colorbar()
plt.title('%s Detection Heatmap' % dataset.name)
plt.axis('off')
plt.tight_layout()
plt.savefig(
    os.path.join(result_path, '%s_heatmap.png' % dataset.name),
    dpi=300
)
plt.close()

# =========================
# Save original + GT + heatmap combined figure
# =========================
rgb_img = hsi_to_rgb(data)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原圖
axes[0].imshow(rgb_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Ground Truth
axes[1].imshow(gt, cmap='gray')
axes[1].set_title('Ground Truth')
axes[1].axis('off')

# Heatmap
im = axes[2].imshow(pr_dm, cmap='jet')
axes[2].set_title('Detection Heatmap')
axes[2].axis('off')

fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(
    os.path.join(result_path, '%s_compare.png' % dataset.name),
    dpi=300
)
plt.close()

print('Complete.')
print('Results are saved in results/.')