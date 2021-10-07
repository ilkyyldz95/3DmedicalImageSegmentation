import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import json
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    EnsureType,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.apps import DecathlonDataset, CrossValidation

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch
import argparse

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
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (crop_size, crop_size, crop_size), 1, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric = [dice_metric.aggregate().item()]
        metric_batch = dice_metric_batch.aggregate()
        for class_idx in range(len(metric_batch)):
            metric.append(metric_batch[class_idx].item())
    dice_metric.reset()
    dice_metric_batch.reset()
    return metric


def train(global_step, train_loader, dice_val_best, global_step_best, dice_val_list_best):
    model.train()
    epoch_loss = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            metric = validation(epoch_iterator_val)  # list of aggregate -> per class
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(metric)
            dice_val = metric[0]
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                dice_val_list_best = metric[1:]
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved At Global Step {}! Current Best Avg. Dice: {} Current Avg. Dice: {} Others: {}"
                        .format(global_step, dice_val_best, dice_val, dice_val_list_best
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Others: {}"
                        .format(global_step, dice_val_best, dice_val, dice_val_list_best
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best, dice_val_list_best

if __name__ == '__main__':
    """
    python unetr_btcv_segmentation_3d.py "./dataset" "Task01_BrainTumour" "./results_segmentation" 4 "./results_ranking/Task01_BrainTumour/recon_lr_0.0001_temp_0.1_best_metric_model.pth" 5 0.0001
    python unetr_btcv_segmentation_3d.py "./dataset" "Task09_Spleen" "./results_segmentation" 2 "./results_ranking/Task09_Spleen/recon_lr_0.0001_temp_0.1_best_metric_model.pth" 5 0.0001
    python unetr_btcv_segmentation_3d.py "./dataset" "abdomenCT" "./results_segmentation" 14 "./results_ranking/abdomenCT/recon_lr_0.0001_temp_0.1_best_metric_model.pth" 5 0.0001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default="./dataset")
    parser.add_argument('dataset_name', type=str, default="abdomenCT")
    parser.add_argument('root_dir', type=str, default="./results_segmentation")
    parser.add_argument('n_classes', type=int, default=14)
    parser.add_argument('pretrained', type=str, default="./results_ranking/abdomenCT/recon_lr_0.0001_temp_0.1_best_metric_model.pth")
    parser.add_argument('n_fold', type=int, default=5)
    parser.add_argument('learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    # Read parameters
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    root_dir = args.root_dir
    n_classes = args.n_classes
    n_fold = args.n_fold
    mode = "train"

    # Set correct root directory
    # if semisupervised
    if "ranking" in args.pretrained:
        root_dir += "_pretrained_ranking"
    elif "contrast" in args.pretrained:
        root_dir += "_pretrained_contrast"
    # add dataset name
    print("Processing dataset", dataset_name)
    root_dir = os.path.join(root_dir, dataset_name)

    # Crop size and input channel size
    if "Task01" in dataset_name:
        crop_size = 128
        add_input_channel = False
    elif "Task09" in dataset_name:
        crop_size = 96
        add_input_channel = True
    else:
        crop_size = 16  # bottleneck features are 2 dimensional
        add_input_channel = True

    # Data transforms, difference for 3D or 4D images
    if add_input_channel:  # 3D image, CT, binary segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
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
                    pixdim=(1.0, 1.0, 1.0),
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
                ToTensord(keys=["image", "label"]),
            ]
        )
        in_channel_size = 1
    else:  # 4D image, MR, multi-class segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
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
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ToTensord(keys=["image", "label"]),
            ]
        )
        in_channel_size = 4

    # Architecture
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR(
        in_channels=in_channel_size,
        out_channels=n_classes-1,
        img_size=(crop_size, crop_size, crop_size),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    # Load pretrained model if exists
    if args.pretrained != "":
        print("Loading pretrained model", args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))

    # Loss and optimizer
    max_iterations = 25000
    eval_num = 500
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )

    # CROSS VALIDATION: load dataset and split
    cvdataset = CrossValidation(
        dataset_cls=DecathlonDataset,
        nfolds=n_fold,
        seed=12345,
        root_dir=data_dir,
        task=dataset_name,
        section="training",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    for fold_idx in range(n_fold):
        # make root directory if does not exist
        root_dir += "_" + str(fold_idx)
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        print("Root directory is {}".format(root_dir))

        # current fold
        val_ds = cvdataset.get_dataset(folds=fold_idx)
        print("Val dataset length: ", len(val_ds))
        train_ds = cvdataset.get_dataset(folds=[fold_idx1
                                        for fold_idx1 in range(n_fold) if fold_idx != fold_idx1])
        print("Train dataset length: ", len(train_ds))

        # Data loader
        train_ds.transform = train_transforms
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
        )
        val_ds.transform = val_transforms
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

        # Training
        if mode == "train":
            global_step = 0
            dice_val_best = 0.0
            dice_val_list_best = 0.0
            global_step_best = 0
            epoch_loss_values = []
            metric_values = []
            while global_step < max_iterations:
                global_step, dice_val_best, global_step_best, dice_val_list_best = train(
                    global_step, train_loader, dice_val_best, global_step_best, dice_val_list_best
                )
            np.save(os.path.join(root_dir, "loss"), epoch_loss_values)
            np.save(os.path.join(root_dir, "metric"), metric_values)

            # Evaluation
            model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
            print(
                "train completed, best_metric: {} ".format(dice_val_best)+
                "best_metric_list: {} ".format(dice_val_list_best)+
                "at iteration: {}".format(global_step_best)
            )

            # Performance visualization
            plt.figure("train", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Iteration Average Loss")
            x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
            y = epoch_loss_values
            plt.xlabel("Iteration")
            plt.plot(x, y)
            plt.subplot(1, 2, 2)
            plt.title("Val Mean Dice")
            x = [eval_num * (i + 1) for i in range(len(metric_values))]
            y = np.array(metric_values[:, 0])
            plt.xlabel("Iteration")
            plt.plot(x, y)
            plt.savefig(os.path.join(root_dir, "train_val.png"))

    # Example visualization
    case_num = 10
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(
            val_inputs, (crop_size, crop_size, crop_size), 4, model, overlap=0.8
        )
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 5], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 5])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(
            torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 5]
        )
        plt.savefig(os.path.join(root_dir, "examples.png"))