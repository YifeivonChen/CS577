import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

############################################
# Section 2: YOLOv1-style model (simplified)
############################################

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU."""
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YoloV1Backbone(nn.Module):
    """
    Approximate YOLOv1 backbone:
    input (B,3,448,448) -> output (B,1024,7,7)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1
            ConvBlock(3,   64, 7, s=2, p=3),
            nn.MaxPool2d(2, 2),

            # Stage 2
            ConvBlock(64, 192, 3, s=1, p=1),
            nn.MaxPool2d(2, 2),

            # Stage 3
            ConvBlock(192, 128, 1),
            ConvBlock(128, 256, 3, p=1),
            ConvBlock(256, 256, 1),
            ConvBlock(256, 512, 3, p=1),
            nn.MaxPool2d(2, 2),

            # Stage 4 (repeated blocks)
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, p=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, p=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, p=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, p=1),
            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, 3, p=1),
            nn.MaxPool2d(2, 2),

            # Stage 5
            ConvBlock(1024, 512, 1),
            ConvBlock(512,  1024, 3, p=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512,  1024, 3, p=1),
            ConvBlock(1024, 1024, 3, p=1),
            ConvBlock(1024, 1024, 3, s=2, p=1),

            # Stage 6
            ConvBlock(1024, 1024, 3, p=1),
            ConvBlock(1024, 1024, 3, p=1),
        )

    def forward(self, x):
        return self.features(x)  # (B, 1024, 7, 7) for 448x448 input


class YoloV1(nn.Module):
    """
    YOLOv1 head on top of backbone.
    """
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.backbone = YoloV1Backbone()
        self.fc1 = nn.Linear(1024 * S * S, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))

    def forward(self, x):
        B = x.size(0)
        x = self.backbone(x)             # (B, 1024, 7, 7)
        x = x.view(B, -1)                # flatten
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = self.fc2(x)                  # (B, 7*7*(B*5+C))
        x = x.view(B, self.S, self.S, self.B * 5 + self.C)
        return x


def yolo_loss(pred, target, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Simplified YOLOv1 loss.
    pred, target: (B, S, S, 5 + C) or (B, S, S, B*5 + C) with B=1 assumed here.
    Layout for B=1 (for simplicity in this example):
        [x, y, w, h, conf, p1, ..., pC]
    """
    # masks for cells with / without object
    obj_mask = target[..., 4] > 0.5
    noobj_mask = ~obj_mask

    # coordinates
    pred_xy = pred[..., 0:2]
    pred_wh = pred[..., 2:4]
    targ_xy = target[..., 0:2]
    targ_wh = target[..., 2:4]

    if obj_mask.any():
        xy_loss = F.mse_loss(pred_xy[obj_mask], targ_xy[obj_mask], reduction="sum")
        pred_sqrt_wh = torch.sign(pred_wh) * torch.sqrt(torch.clamp(pred_wh.abs(), min=1e-6))
        targ_sqrt_wh = torch.sqrt(torch.clamp(targ_wh, min=1e-6))
        wh_loss = F.mse_loss(pred_sqrt_wh[obj_mask], targ_sqrt_wh[obj_mask], reduction="sum")
    else:
        xy_loss = torch.tensor(0.0, device=pred.device)
        wh_loss = torch.tensor(0.0, device=pred.device)

    # confidence
    pred_conf = pred[..., 4]
    targ_conf = target[..., 4]
    if obj_mask.any():
        conf_obj = F.mse_loss(pred_conf[obj_mask], targ_conf[obj_mask], reduction="sum")
    else:
        conf_obj = torch.tensor(0.0, device=pred.device)
    if noobj_mask.any():
        conf_no = F.mse_loss(pred_conf[noobj_mask], targ_conf[noobj_mask], reduction="sum")
    else:
        conf_no = torch.tensor(0.0, device=pred.device)

    # classification (if C>0)
    if pred.size(-1) > 5:
        pred_cls = pred[..., 5:]
        targ_cls = target[..., 5:]
        if obj_mask.any():
            cls_loss = F.mse_loss(pred_cls[obj_mask], targ_cls[obj_mask], reduction="sum")
        else:
            cls_loss = torch.tensor(0.0, device=pred.device)
    else:
        cls_loss = torch.tensor(0.0, device=pred.device)

    loss = lambda_coord * (xy_loss + wh_loss) + conf_obj + lambda_noobj * conf_no + cls_loss
    return loss / pred.size(0)


############################################
# Section 4: MWE-CPU (Tiny YOLO + synthetic data)
############################################

class SyntheticSquareDataset(Dataset):
    """
    Synthetic dataset: 64x64 black image with one white rectangle.
    Target: (S,S,6) tensor per sample.
    """
    def __init__(self, num_samples=2000, image_size=64, S=4):
        self.num_samples = num_samples
        self.image_size = image_size
        self.S = S
        self.cell = image_size // S

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        H = W = self.image_size
        img = torch.zeros(3, H, W, dtype=torch.float32)

        # random rectangle
        w = random.randint(H // 8, H // 2)
        h = random.randint(H // 8, H // 2)
        x1 = random.randint(0, W - w - 1)
        y1 = random.randint(0, H - h - 1)
        x2, y2 = x1 + w, y1 + h

        img[:, y1:y2, x1:x2] = 1.0

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        gx = int(cx // self.cell)
        gy = int(cy // self.cell)

        target = torch.zeros(self.S, self.S, 6, dtype=torch.float32)
        tx = (cx - gx * self.cell) / self.cell
        ty = (cy - gy * self.cell) / self.cell
        tw = w / W
        th = h / H

        target[gy, gx, 0] = tx
        target[gy, gx, 1] = ty
        target[gy, gx, 2] = tw
        target[gy, gx, 3] = th
        target[gy, gx, 4] = 1.0  # conf
        target[gy, gx, 5] = 1.0  # class (single-class)

        return img, target


class TinyYolo(nn.Module):
    """
    Tiny YOLO for CPU MWE.
    Input: (B,3,64,64) -> Output: (B,S,S,6)
    """
    def __init__(self, S=4):
        super().__init__()
        self.S = S
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, S * S * 6)

    def forward(self, x):
        B = x.size(0)
        x = self.features(x)
        x = x.view(B, -1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.fc2(x)
        return x.view(B, self.S, self.S, 6)


def tiny_yolo_loss(pred, target, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Loss for tiny YOLO MWE.
    pred, target: (B, S, S, 6) with layout [x,y,w,h,conf,cls]
    """
    obj = target[..., 4] > 0.5
    noobj = ~obj

    xy_loss = ((pred[..., 0:2][obj] - target[..., 0:2][obj])**2).sum()

    pred_wh = torch.sqrt(torch.clamp(pred[..., 2:4], min=1e-6))
    targ_wh = torch.sqrt(torch.clamp(target[..., 2:4], min=1e-6))
    wh_loss = ((pred_wh[obj] - targ_wh[obj])**2).sum()

    conf_obj = ((pred[..., 4][obj] - target[..., 4][obj])**2).sum()
    conf_no = ((pred[..., 4][noobj] - target[..., 4][noobj])**2).sum()

    cls_loss = ((pred[..., 5][obj] - target[..., 5][obj])**2).sum()

    return lambda_coord * (xy_loss + wh_loss) + conf_obj + lambda_noobj * conf_no + cls_loss


############################################
# Example training loops
############################################

def train_tiny_yolo_cpu(num_epochs=10, S=4, batch_size=32):
    dataset = SyntheticSquareDataset(num_samples=2000, image_size=64, S=S)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    model = TinyYolo(S=S).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = tiny_yolo_loss(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch}: avg_loss={epoch_loss:.4f}")

    return model


if __name__ == "__main__":
    # Quick sanity check for Tiny YOLO MWE
    train_tiny_yolo_cpu(num_epochs=3, S=4, batch_size=32)
