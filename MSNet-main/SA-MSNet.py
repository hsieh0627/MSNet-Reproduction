from dataset import coast
from dataset import gulfport
from model import MSNet

import matplotlib.pyplot as plt
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
import os
import random

import select_bands
import utils
import metric

from SeT import (
    TotalLoss,
    Mask,
    separation_training
)


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Helper functions
# =========================================================
def normalize_img(img, p_low=2, p_high=98):
    """
    Percentile normalization.
    比單純 min-max 更適合顯示高光譜影像與 heatmap。
    """
    img = img.astype(np.float32)

    low, high = np.percentile(img, [p_low, p_high])
    img = np.clip(img, low, high)

    if high > low:
        img = (img - low) / (high - low)
    else:
        img = np.zeros_like(img)

    return img


def hsi_to_rgb(data):
    """
    將 HSI cube 轉成可視覺化的假 RGB 圖。
    data shape: H × W × B
    """
    bands = data.shape[2]

    if bands >= 3:
        b1 = 0
        b2 = bands // 2
        b3 = bands - 1
        rgb = np.stack(
            [data[:, :, b1], data[:, :, b2], data[:, :, b3]],
            axis=-1
        )
    else:
        gray = data[:, :, 0]
        rgb = np.stack([gray, gray, gray], axis=-1)

    rgb = normalize_img(rgb)
    return rgb


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# =========================================================
# Spectral Attention Module
# =========================================================
class SpectralAttention(nn.Module):
    """
    Spectral Attention for HSI.

    Input:
        x: H × W × C

    C means selected spectral bands after OPBS.
    This module learns a weight for each selected band.
    """
    def __init__(self, channels, reduction=8):
        super(SpectralAttention, self).__init__()

        hidden_dim = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Case 1: H × W × C
        if x.dim() == 3:
            z = torch.mean(x, dim=(0, 1))      # C
            w = self.fc(z).view(1, 1, -1)      # 1 × 1 × C
            return x * w

        # Case 2: N × H × W × C
        elif x.dim() == 4:
            z = torch.mean(x, dim=(1, 2))      # N × C
            w = self.fc(z).view(x.shape[0], 1, 1, -1)
            return x * w

        else:
            raise ValueError(
                "Unsupported input shape for SpectralAttention: {}".format(x.shape)
            )

    def get_attention_weights(self, x):
        """
        用於輸出 attention weight。
        x shape: H × W × C
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                z = torch.mean(x, dim=(0, 1))
                w = self.fc(z)
                return w.detach().cpu().numpy()
            elif x.dim() == 4:
                z = torch.mean(x, dim=(1, 2))
                w = self.fc(z)
                return w.detach().cpu().numpy()
            else:
                raise ValueError(
                    "Unsupported input shape for attention weights: {}".format(x.shape)
                )


# =========================================================
# SA-MSNet: OPBS + Spectral Attention + MSNet
# =========================================================
class SA_MSNet(nn.Module):
    """
    Spectral Attention Enhanced MSNet.

    Architecture:
        OPBS selected HSI
            ↓
        Spectral Attention
            ↓
        MSNet
            ↓
        SeT
    """
    def __init__(self, num_layers, shape, reduction=8):
        super(SA_MSNet, self).__init__()

        self.name = "SA_MSNet"

        channels = shape[2]

        self.spectral_attention = SpectralAttention(
            channels=channels,
            reduction=reduction
        )

        self.msnet = MSNet(
            num_layers=num_layers,
            shape=shape
        )

    def forward(self, x):
        x = self.spectral_attention(x)
        x = self.msnet(x)
        return x


# =========================================================
# Settings
# =========================================================
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lmda = 1e-3
num_bs = 64
num_layers = 3
lr = 1e-3
epochs = 150
output_iter = 5
max_iter = 10
data_norm = True

# 改成 SA_MSNet
Net = SA_MSNet

net_kwargs = dict()
net_kwargs['num_layers'] = num_layers
net_kwargs['reduction'] = 8


# =========================================================
# Dataset selection
# =========================================================
dataset_dict = {
    "coast": coast,
    "gulfport": gulfport,
}

# 想跑哪個資料集就改這裡
dataset_name = "coast"
dataset = dataset_dict[dataset_name]


# =========================================================
# Load data
# =========================================================
data, gt = dataset.get_data()

data = data.astype(np.float32)
gt = gt.astype(bool)

rows, cols, bands = data.shape

net_kwargs['shape'] = (rows, cols, num_bs)

print("=" * 60)
print("Detecting on %s..." % dataset.name)
print("Data shape:", data.shape)
print("GT shape:", gt.shape)
print("Device:", device)
print("Model: SA-MSNet")
print("=" * 60)


# =========================================================
# Preprocessing: OPBS + Z-score normalization
# =========================================================
print("Selecting bands by OPBS...")
band_idx = select_bands.OPBS(data, num_bs)

data_bs = data[:, :, band_idx]

if data_norm:
    print("Applying Z-score normalization...")
    data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

data_bs = data_bs.astype(np.float32)


# =========================================================
# Load model
# =========================================================
model = Net(**net_kwargs).to(device).float()

loss = TotalLoss(lmda, device)
mask = Mask((rows, cols), device)
optimizer = Adam(model.parameters(), lr=lr)


# =========================================================
# Separation Training
# =========================================================
print("Start separation training...")

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

pr_dm = to_numpy(pr_dm)
gt = to_numpy(gt)


# =========================================================
# Save result directory
# =========================================================
result_path = os.path.join('results', model.name, dataset.name)
os.makedirs(result_path, exist_ok=True)


# =========================================================
# Save selected band indices
# =========================================================
np.save(
    os.path.join(result_path, '%s_selected_band_indices.npy' % dataset.name),
    band_idx
)


# =========================================================
# Save spectral attention weights
# =========================================================
print("Saving spectral attention weights...")

with torch.no_grad():
    x_tmp = torch.from_numpy(data_bs).to(device).float()
    att_weights = model.spectral_attention.get_attention_weights(x_tmp)

np.save(
    os.path.join(result_path, '%s_spectral_attention_weights.npy' % dataset.name),
    att_weights
)

plt.figure(figsize=(9, 4))
plt.plot(att_weights, marker='o')
plt.xlabel('Selected Band Index')
plt.ylabel('Attention Weight')
plt.title('%s Spectral Attention Weights' % dataset.name)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(result_path, '%s_spectral_attention_weights.png' % dataset.name),
    dpi=300
)
plt.close()


# =========================================================
# RX baseline
# =========================================================
print("Computing RX baseline...")

rx_dm = utils.rx(data)
rx_dm = to_numpy(rx_dm)


# =========================================================
# AUC calculation
# =========================================================
fpr_rx, tpr_rx, rx_auc = metric.roc_auc(rx_dm, gt)
fpr_pr, tpr_pr, pr_auc = metric.roc_auc(pr_dm, gt)

print("RX AUC       : %.6f" % rx_auc)
print("SA-MSNet AUC : %.6f" % pr_auc)


# =========================================================
# Save raw maps
# =========================================================
np.save(
    os.path.join(result_path, '%s_sa_msnet_detection_map.npy' % dataset.name),
    pr_dm
)

np.save(
    os.path.join(result_path, '%s_rx_detection_map.npy' % dataset.name),
    rx_dm
)

np.save(
    os.path.join(result_path, '%s_ground_truth.npy' % dataset.name),
    gt
)


# =========================================================
# Save metrics txt
# =========================================================
with open(os.path.join(result_path, '%s_metrics.txt' % dataset.name), 'w') as f:
    f.write('Dataset: %s\n' % dataset.name)
    f.write('Model: %s\n' % model.name)
    f.write('OPBS selected bands: %d\n' % num_bs)
    f.write('Spectral attention reduction: %d\n' % net_kwargs['reduction'])
    f.write('RX AUC: %.6f\n' % rx_auc)
    f.write('SA-MSNet+SeT AUC: %.6f\n' % pr_auc)
    f.write('Best AUC in history: %.6f\n' % max(history))
    f.write('Best Iteration: %d\n' % (np.argmax(history) + 1))
    f.write('Best Epoch: %d\n' % ((np.argmax(history) + 1) * epochs))


# =========================================================
# ROC curve -> PNG
# =========================================================
plt.figure(figsize=(6, 5))
plt.plot(fpr_rx, tpr_rx, label='RX: %.4f' % rx_auc)
plt.plot(
    fpr_pr,
    tpr_pr,
    label='%s+SeT: %.4f' % (model.name, pr_auc),
    c='black',
    alpha=0.7
)

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


# =========================================================
# AUC history -> PNG
# =========================================================
iters = [(_ + 1) * epochs for _ in range(max_iter)]

plt.figure(figsize=(6, 5))
plt.xticks(iters)
plt.plot(iters, history, marker='o')

plt.scatter(
    [output_iter * epochs],
    [history[output_iter - 1]],
    marker='o',
    edgecolors='black',
    facecolors='white',
    label='Stop',
    zorder=10
)

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


# =========================================================
# Normalize detection maps for visualization
# =========================================================
pr_dm_norm = normalize_img(pr_dm)
rx_dm_norm = normalize_img(rx_dm)


# =========================================================
# Save SA-MSNet heatmap only -> PNG
# =========================================================
plt.figure(figsize=(6, 5))
plt.imshow(pr_dm_norm, cmap='jet')
plt.colorbar()
plt.title('%s SA-MSNet Detection Heatmap' % dataset.name)
plt.axis('off')
plt.tight_layout()

plt.savefig(
    os.path.join(result_path, '%s_sa_msnet_heatmap.png' % dataset.name),
    dpi=300
)
plt.close()


# =========================================================
# Save RX heatmap only -> PNG
# =========================================================
plt.figure(figsize=(6, 5))
plt.imshow(rx_dm_norm, cmap='jet')
plt.colorbar()
plt.title('%s RX Detection Heatmap' % dataset.name)
plt.axis('off')
plt.tight_layout()

plt.savefig(
    os.path.join(result_path, '%s_rx_heatmap.png' % dataset.name),
    dpi=300
)
plt.close()


# =========================================================
# Save binary anomaly map
# =========================================================
threshold = np.percentile(pr_dm, 99)
binary_map = pr_dm >= threshold

plt.figure(figsize=(6, 5))
plt.imshow(binary_map, cmap='gray')
plt.title('%s Binary Anomaly Map, Threshold=P99' % dataset.name)
plt.axis('off')
plt.tight_layout()

plt.savefig(
    os.path.join(result_path, '%s_binary_map.png' % dataset.name),
    dpi=300
)
plt.close()


# =========================================================
# Save original + GT + RX heatmap + SA-MSNet heatmap
# =========================================================
rgb_img = hsi_to_rgb(data)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(rgb_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(gt, cmap='gray')
axes[1].set_title('Ground Truth')
axes[1].axis('off')

axes[2].imshow(rx_dm_norm, cmap='jet')
axes[2].set_title('RX Heatmap\nAUC = %.4f' % rx_auc)
axes[2].axis('off')

im = axes[3].imshow(pr_dm_norm, cmap='jet')
axes[3].set_title('SA-MSNet Heatmap\nAUC = %.4f' % pr_auc)
axes[3].axis('off')

fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.tight_layout()

plt.savefig(
    os.path.join(result_path, '%s_compare.png' % dataset.name),
    dpi=300
)
plt.close()


# =========================================================
# Save overlay image
# =========================================================
plt.figure(figsize=(6, 5))
plt.imshow(rgb_img)
plt.imshow(pr_dm_norm, cmap='jet', alpha=0.45)
plt.title('%s SA-MSNet Overlay Heatmap' % dataset.name)
plt.axis('off')
plt.tight_layout()

plt.savefig(
    os.path.join(result_path, '%s_overlay.png' % dataset.name),
    dpi=300
)
plt.close()


print("=" * 60)
print("Complete.")
print("Results are saved in:", result_path)
print("=" * 60)