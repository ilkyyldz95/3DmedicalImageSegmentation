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
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
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
    label 2 is the necrotic and non-enhancing tumor core
    label 3 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # background
            result.append(d[key] == 0)
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 3 is ET
            result.append(d[key] == 3)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (crop_size, crop_size, crop_size), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = [dice_metric.aggregate().item()]
            epoch_iterator_val.set_description\
                ("Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, max_iterations, dice[0]))
            # per class evaluation
            dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            dice_batch = dice_metric_batch.aggregate()
            for class_idx in range(len(dice_batch)):
                dice.append(dice_batch[class_idx].item())
            dice_vals.append(dice)
        dice_metric.reset()
        dice_metric_batch.reset()
    dice_vals = np.array(dice_vals)
    mean_dice_val = np.mean(dice_vals, 0)
    return mean_dice_val

def validation_all_metrics(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    precision_vals = list()
    recall_vals = list()
    hsd_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (crop_size, crop_size, crop_size), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]

            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = [dice_metric.aggregate().item()]

            precision_metric(y_pred=val_output_convert, y=val_labels_convert)
            precision = [precision_metric.aggregate()[0].item()]

            recall_metric(y_pred=val_output_convert, y=val_labels_convert)
            recall = [recall_metric.aggregate()[0].item()]

            hsd_metric(y_pred=val_output_convert, y=val_labels_convert)
            hsd = [hsd_metric.aggregate().item()]

            epoch_iterator_val.set_description\
                ("Validate (%d / %d Steps) (dice=%2.5f) (prec=%2.5f) (rec=%2.5f) (hsd=%2.5f)" \
                 % (global_step, max_iterations, dice[0], precision[0], recall[0], hsd[0]))

            # per class evaluation
            dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            dice_batch = dice_metric_batch.aggregate()
            for class_idx in range(len(dice_batch)):
                dice.append(dice_batch[class_idx].item())
            dice_vals.append(dice)

            precision_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            precision_batch = precision_metric_batch.aggregate()[0]
            for class_idx in range(len(precision_batch)):
                precision.append(precision_batch[class_idx].item())
            precision_vals.append(precision)

            recall_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            recall_batch = recall_metric_batch.aggregate()[0]
            for class_idx in range(len(recall_batch)):
                recall.append(recall_batch[class_idx].item())
            recall_vals.append(recall)

            hsd_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            hsd_batch = hsd_metric_batch.aggregate()
            for class_idx in range(len(hsd_batch)):
                hsd.append(hsd_batch[class_idx].item())
            hsd_vals.append(hsd)
        dice_metric.reset()
        dice_metric_batch.reset()
        precision_metric.reset()
        precision_metric_batch.reset()
        recall_metric.reset()
        recall_metric_batch.reset()
        hsd_metric.reset()
        hsd_metric_batch.reset()
    dice_vals = np.array(dice_vals)
    precision_vals = np.array(precision_vals)
    recall_vals = np.array(recall_vals)
    hsd_vals = np.array(hsd_vals)
    mean_dice_val = np.mean(dice_vals, 0)
    mean_precision_val = np.mean(precision_vals, 0)
    mean_recall_val = np.mean(recall_vals, 0)
    mean_hsd_val = np.mean(hsd_vals, 0)
    return mean_dice_val, mean_precision_val, mean_recall_val, mean_hsd_val

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
            dice_values_list.append(metric)
            dice_val = metric[0]
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                dice_val_list_best = metric[1:]
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved At Global Step {}! Current Best Avg. Dice: {} Current Avg. Dice: {} Per class: {}"
                        .format(global_step, dice_val_best, dice_val, dice_val_list_best
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Per class: {}"
                        .format(dice_val_best, dice_val, dice_val_list_best
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
        case_num = 0
        slice_num = 80

        # Loss
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        # Metrics
        post_label = AsDiscrete(to_onehot=True, n_classes=n_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=n_classes)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        precision_metric = ConfusionMatrixMetric(include_background=False, reduction="mean", get_not_nans=False,
                                          metric_name="precision")
        precision_metric_batch = ConfusionMatrixMetric(include_background=False, reduction="mean_batch", get_not_nans=False,
                                                metric_name="precision")
        recall_metric = ConfusionMatrixMetric(include_background=False, reduction="mean", get_not_nans=False,
                                          metric_name="sensitivity")
        recall_metric_batch = ConfusionMatrixMetric(include_background=False, reduction="mean_batch", get_not_nans=False,
                                                metric_name="sensitivity")
        hsd_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)
        hsd_metric_batch = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

    else:  # 4D image, MR, multi-class segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                #AddChanneld(keys="label"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                #CropForegroundd(keys=["image", "label"], source_key="image"),
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
                #AddChanneld(keys="label"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                #CropForegroundd(keys=["image", "label"], source_key="image"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ToTensord(keys=["image", "label"]),
            ]
        )
        in_channel_size = 4
        case_num = 0
        slice_num = 60

        # Loss
        loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
        # Metrics
        post_label = AsDiscrete(to_onehot=False, n_classes=n_classes)
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
        precision_metric = ConfusionMatrixMetric(include_background=True, reduction="mean", get_not_nans=False,
                                          metric_name="precision")
        precision_metric_batch = ConfusionMatrixMetric(include_background=True, reduction="mean_batch", get_not_nans=False,
                                                metric_name="precision")
        recall_metric = ConfusionMatrixMetric(include_background=True, reduction="mean", get_not_nans=False,
                                       metric_name="sensitivity")
        recall_metric_batch = ConfusionMatrixMetric(include_background=True, reduction="mean_batch", get_not_nans=False,
                                             metric_name="sensitivity")
        hsd_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
        hsd_metric_batch = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    # Architecture
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR(
        in_channels=in_channel_size,
        out_channels=n_classes,
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

    # Optimizer
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

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

        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        img_shape = img.shape
        label_shape = label.shape
        print(f"image shape: {img_shape}, label shape: {label_shape}")
        plt.figure("image", (24, 6))
        for i in range(img_shape[0]):
            plt.subplot(1, img_shape[0], i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(img[i, :, :, slice_num].detach().cpu(), cmap="gray")
        plt.savefig(os.path.join(root_dir, "exampleimage.png"))
        # also visualize the 3 channels label corresponding to this image
        plt.figure("label", (18, 6))
        for i in range(label_shape[0]):
            plt.subplot(1, label_shape[0], i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(label[i, :, :, slice_num].detach().cpu())
        plt.savefig(os.path.join(root_dir, "examplelabel.png"))

        # Training
        if mode == "train":
            global_step = 0
            max_iterations = 25000
            eval_num = 500

            dice_val_best = 0.0
            dice_val_list_best = 0.0
            global_step_best = 0
            epoch_loss_values = []
            dice_values_list = []
            # checkpoint if exists
            if os.path.exists(os.path.join(root_dir, "best_metric_model.pth")):
                print("Loading Model Saved At Global Step {}!".format(global_step))
                model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
            while global_step < max_iterations:
                global_step, dice_val_best, global_step_best, dice_val_list_best = train(
                    global_step, train_loader, dice_val_best, global_step_best, dice_val_list_best
                )

            # Evaluation
            model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
            # calculate all metrics
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X) (prec=X.X) (rec=X.X) (hsd=X.X)", dynamic_ncols=True
            )
            mean_dice_val, mean_precision_val, mean_recall_val, mean_hsd_val = validation_all_metrics(
                epoch_iterator_val)  # list of aggregate -> per class
            # save dice and loss from all steps and all final metrics
            np.save(os.path.join(root_dir, "loss"), epoch_loss_values)
            np.save(os.path.join(root_dir, "dice_values_list"), dice_values_list)
            np.save(os.path.join(root_dir, "precision_values"), mean_precision_val)
            np.save(os.path.join(root_dir, "recall_values"), mean_recall_val)
            np.save(os.path.join(root_dir, "hsd_values"), mean_hsd_val)
            print(
                "train completed, best dice: {} ".format(dice_val_best) +
                "per class: {} ".format(dice_val_list_best) +
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
            x = [eval_num * (i + 1) for i in range(len(dice_values_list))]
            y = np.array(dice_values_list)[:, 0]
            plt.xlabel("Iteration")
            plt.plot(x, y)
            plt.savefig(os.path.join(root_dir, "train_val.png"))

        # Example visualization
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
            plt.imshow(val_inputs.cpu().numpy()[case_num, 0, :, :, slice_num], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(val_labels.cpu().numpy()[case_num, 0, :, :, slice_num])
            plt.subplot(1, 3, 3)
            plt.title("output")
            plt.imshow(
                torch.argmax(val_outputs, dim=1).detach().cpu()[case_num, :, :, slice_num]
            )
            plt.savefig(os.path.join(root_dir, "examples.png"))