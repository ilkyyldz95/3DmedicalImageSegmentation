import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch
torch.autograd.set_detect_anomaly(True)

# Tuned parameters based on dataset
train_size = 50
val_size = 33
n_classes = 14
root_dir = "./results_ranking/"
data_dir = "./dataset/"
learning_rate = 1e-4
neighbour_size = 4
crop_size = 16
"""
- keys for image and label
>>> labels = os.listdir("labelsTr")
>>> labels.sort()
>>> labels = os.listdir("imagesTr")
>>> labels = os.listdir("labelsTr")
>>> labels.sort()
>>> images = os.listdir("imagesTr")
>>> images.sort()
>>> json_dict["training"] = []
>>> for i in range(50):
    json_dict["training"].append({"image": "imagesTr/"+images[i], "label": "labelsTr/" + labels[i]})
ds = Dataset(json_dict)
- image size as crop_size / ROI, may resize to this default shape
image shape: torch.Size([n_img_channels, img_dim_x, img_dim_y, img_dim_z])
label shape: torch.Size([n_seg_classes, img_dim_x, img_dim_y, img_dim_z])
n_seg_classes = 2 (edema / tumor core)
n_img_channels, img_dim_x, img_dim_y, img_dim_z = 1, 91, 109, 91
"""

def BTLoss(y_pred, optimizer):
    """
    y_pred: (batch, n_labels, x, y, z)
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    comp_pred:si-sj
    """
    print(y_pred.shape)
    cum_loss = 0
    # Axis 1
    similar_pred = y_pred.clone()[:, :, :neighbour_size, :, :] \
                   - y_pred.clone()[:, :, neighbour_size:2*neighbour_size, :, :]
    dissimilar_pred = y_pred.clone()[:, :, :neighbour_size, :, :] - y_pred.clone()[:, :, -neighbour_size:, :, :]
    print(similar_pred.shape, dissimilar_pred.shape)
    comp_pred = similar_pred - dissimilar_pred
    loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward(retain_graph=True)
    cum_loss += loss.item()
    similar_pred = y_pred.clone()[:, :, -neighbour_size:, :, :] \
                   - y_pred.clone()[:, :, -2 * neighbour_size:-neighbour_size, :, :]
    dissimilar_pred = y_pred.clone()[:, :, -neighbour_size:, :, :] - y_pred.clone()[:, :, :neighbour_size, :, :]
    print(similar_pred.shape, dissimilar_pred.shape)
    comp_pred = similar_pred - dissimilar_pred
    loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward(retain_graph=True)
    cum_loss += loss.item()
    # Axis 2
    similar_pred = y_pred.clone()[:, :, :, :neighbour_size, :]\
                   - y_pred.clone()[:, :, :, neighbour_size:2 * neighbour_size, :]
    dissimilar_pred = y_pred.clone()[:, :, :, :neighbour_size, :] - y_pred.clone()[:, :, :, -neighbour_size:, :]
    print(similar_pred.shape, dissimilar_pred.shape)
    comp_pred = similar_pred - dissimilar_pred
    loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward(retain_graph=True)
    cum_loss += loss.item()
    similar_pred = y_pred.clone()[:, :, :, -neighbour_size:, :] \
                   - y_pred.clone()[:, :, :, -2 * neighbour_size:-neighbour_size, :]
    dissimilar_pred = y_pred.clone()[:, :, :, -neighbour_size:, :] - y_pred.clone()[:, :, :, :neighbour_size, :]
    print(similar_pred.shape, dissimilar_pred.shape)
    comp_pred = similar_pred - dissimilar_pred
    loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward(retain_graph=True)
    cum_loss += loss.item()
    # Axis 3
    similar_pred = y_pred.clone()[:, :, :, :, :neighbour_size] \
                   - y_pred.clone()[:, :, :, :, neighbour_size:2 * neighbour_size]
    dissimilar_pred = y_pred.clone()[:, :, :, :, :neighbour_size] - y_pred.clone()[:, :, :, :, -neighbour_size:]
    print(similar_pred.shape, dissimilar_pred.shape)
    comp_pred = similar_pred - dissimilar_pred
    loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward(retain_graph=True)
    cum_loss += loss.item()
    similar_pred = y_pred.clone()[:, :, :, :, -neighbour_size:] \
                   - y_pred.clone()[:, :, :, :, -2 * neighbour_size:-neighbour_size]
    dissimilar_pred = y_pred.clone()[:, :, :, :, -neighbour_size:] - y_pred.clone()[:, :, :, :, :neighbour_size]
    print(similar_pred.shape, dissimilar_pred.shape)
    comp_pred = similar_pred - dissimilar_pred
    loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    cum_loss += loss.item()
    return cum_loss

# Data transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(crop_size, crop_size, crop_size),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

split_JSON = "dataset_0.json"
datasets = data_dir + split_JSON
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=train_size,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=val_size, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

case_num = 0
img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
model = UNETR(
    in_channels=1,
    out_channels=n_classes,
    img_size=(crop_size, crop_size, crop_size),
    feature_size=crop_size,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def train(global_step, train_loader):
    model.train()
    epoch_ranking_loss = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        ranking_loss = BTLoss(logit_map, optimizer)
        epoch_ranking_loss += ranking_loss
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (ranking loss=%2.5f)" % (global_step, max_iterations, ranking_loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_ranking_loss /= step
            epoch_ranking_loss_values.append(epoch_ranking_loss)
            torch.save(
                model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
            )
            print(
                "Model Was Saved At Global Step {}!".format(global_step)
            )
        global_step += 1
    return global_step


max_iterations = 25000
global_step = 0
epoch_ranking_loss_values = []
while global_step < max_iterations:
    global_step = train(global_step, train_loader)

# Evaluation
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

# Performance visualization
plt.figure("train", (12, 6))
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_ranking_loss_values))]
y = epoch_ranking_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig(os.path.join(root_dir, "train.png"))
