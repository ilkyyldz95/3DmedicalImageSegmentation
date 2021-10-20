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
from monai.networks.nets import UNETR, SegResNet, UNet
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
data_dir = "/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/ncs2021_abstract"
label_sub_dir = "lesion_segmentations/Affine"
image_sub_dir = "t2flair/Affine"
- keys for image and label
labels = os.listdir(os.path.join(data_dir, label_sub_dir))
labels.sort()
images = os.listdir(os.path.join(data_dir, image_sub_dir))
images.sort()
n_train = int(len(labels)*0.8)
json_dict["training"] = []
for i in range(n_train):
    json_dict["training"].append({"image": os.path.join(image_sub_dir, images[i]), "label": os.path.join(label_sub_dir, labels[i])})
json_dict["validation"] = []
for i in np.arange(n_train, len(labels)):
    json_dict["validation"].append({"image": os.path.join(image_sub_dir, images[i]), "label": os.path.join(label_sub_dir, labels[i])})
- save json as dataset_0.json to the same path as images and labels

- image size as crop_size / ROI, may resize to this default shape
image shape: torch.Size([n_img_channels, img_dim_x, img_dim_y, img_dim_z])
label shape: torch.Size([n_seg_classes, img_dim_x, img_dim_y, img_dim_z])
n_seg_classes = 2 (edema / tumor core)
n_img_channels, img_dim_x, img_dim_y, img_dim_z = 1, 182, 218, 182
"""
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

def train(global_step, train_loader, dice_val_best, global_step_best, dice_val_list_best, model_save_prefix):
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
                    model.state_dict(), os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")
                )
                print(
                    "Model Was Saved At Global Step {}! Current Best Avg. Dice: {} Current Avg. Dice: {} Per class: {}"
                        .format(global_step, dice_val_best, dice_val, dice_val_list_best
                    )
                )
                logger_file.write("Model Was Saved At Global Step {}! Current Best Avg. Dice: {} "
                                  "Current Avg. Dice: {} Per class: {} \n"
                                .format(global_step, dice_val_best, dice_val, dice_val_list_best))
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Per class: {}"
                        .format(dice_val_best, dice_val, dice_val_list_best
                    )
                )
                logger_file.write("Model Was Not Saved ! Current Best Avg. Dice: {} "
                                  "Current Avg. Dice: {} Per class: {} \n"
                                  .format(dice_val_best, dice_val, dice_val_list_best))
        global_step += 1
    return global_step, dice_val_best, global_step_best, dice_val_list_best

if __name__ == '__main__':
    """
    python unetr_t2_lesion_segmentation_3d.py "./dataset/t2_lesion_segmentation" "./results_segmentation/t2_lesion_segmentation" 2 "train" "unetr" 0.001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default="")
    parser.add_argument('root_dir', type=str, default="./results_segmentation/t2_lesion_segmentation")
    parser.add_argument('n_classes', type=int, default=2)
    parser.add_argument('mode', type=str, default="train")
    parser.add_argument('architecture', type=str, default="unetr")
    parser.add_argument('learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    # Read parameters
    data_dir = args.data_dir
    root_dir = args.root_dir
    n_classes = args.n_classes
    mode = args.mode
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    print("Data directory is {}".format(data_dir))
    print("Root directory is {}".format(root_dir))

    # Crop size and input channel size
    crop_size = 48
    in_channel_size = 1

    # Data transforms, 3D image
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
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # Loss
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    # Metrics
    post_label = AsDiscrete(to_onehot=True, n_classes=n_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=n_classes)
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
    if args.architecture == "unet":
        model = UNet(
            dimensions=3,
            in_channels=in_channel_size,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    elif args.architecture == "segresnet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=in_channel_size,
            out_channels=n_classes,
            dropout_prob=0.2,
        ).to(device)
    else:
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

    # Optimizer
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Data loader
    split_JSON = "dataset_0_bin.json"
    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist, transform=train_transforms, cache_rate=1.0, num_workers=8,
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print("Val dataset length: ", len(val_ds))
    print("Train dataset length: ", len(train_ds))
    img = val_ds[0]["image"]
    label = val_ds[0]["label"]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print("Unique labels", np.unique(label.cpu().numpy()))

    for case_num in range(len(val_ds)):
        val_inputs = val_ds[case_num]["image"]
        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        val_label = val_ds[case_num]["label"]
        for slice_num in range(val_inputs.shape[-1]):
            unique_classes, counts = np.unique(val_label[0, :, :, slice_num].numpy(), return_counts=True)
            if len(unique_classes) < n_classes:
                continue
            class_ratio = counts[1] / counts[0]
            # Plot only if the ground-truth is large enough
            if class_ratio < 0.05:
                continue
            plt.figure("example_input_{}_{}".format(img_name, slice_num), (18, 6))
            plt.subplot(1, 2, 1)
            plt.title("image")
            plt.imshow(val_inputs[0, :, :, slice_num].detach().cpu(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("with label")
            plt.imshow(val_inputs[0, :, :, slice_num].detach().cpu(), 'gray', interpolation='none')
            plt.imshow(val_label[0, :, :, slice_num], 'magma', interpolation='none', alpha=0.5)
            plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.savefig(os.path.join(root_dir, "example_input_{}_{}.pdf".format(img_name, slice_num)))
            break

    # Training
    model_save_prefix = "lr_{}_arch_{}".format(args.learning_rate, args.architecture)
    if mode == "train":
        global_step = 0
        max_iterations = 25000
        eval_num = 50

        dice_val_best = 0.0
        dice_val_list_best = 0.0
        global_step_best = 0
        epoch_loss_values = []
        dice_values_list = []
        # checkpoint if exists
        if os.path.exists(os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")):
            print("Loading Model Saved At Global Step {}!".format(global_step))
            model.load_state_dict(torch.load(os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")))
        while global_step < max_iterations:
            logger_file = open(os.path.join(root_dir, model_save_prefix + "_logger.txt"), "a")
            global_step, dice_val_best, global_step_best, dice_val_list_best = train(
                global_step, train_loader, dice_val_best, global_step_best, dice_val_list_best, model_save_prefix
            )
            logger_file.close()

        # Evaluation
        model.load_state_dict(torch.load(os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")))
        # calculate all metrics
        epoch_iterator_val = tqdm(
            val_loader, desc="Validate (X / X Steps) (dice=X.X) (prec=X.X) (rec=X.X) (hsd=X.X)", dynamic_ncols=True
        )
        mean_dice_val, mean_precision_val, mean_recall_val, mean_hsd_val = validation_all_metrics(
            epoch_iterator_val)  # list of aggregate -> per class
        # save dice and loss from all steps and all final metrics
        np.save(os.path.join(root_dir, model_save_prefix + "_loss"), epoch_loss_values)
        np.save(os.path.join(root_dir, model_save_prefix + "_dice_values_list"), mean_dice_val)
        np.save(os.path.join(root_dir, model_save_prefix + "_precision_values"), mean_precision_val)
        np.save(os.path.join(root_dir, model_save_prefix + "_recall_values"), mean_recall_val)
        np.save(os.path.join(root_dir, model_save_prefix + "_hsd_values"), mean_hsd_val)
        print(
            "train completed, best dice: {} ".format(dice_val_best) +
            "per class: {} ".format(dice_val_list_best) +
            "at iteration: {}".format(global_step_best)
        )
        logger_file = open(os.path.join(root_dir, model_save_prefix + "_logger.txt"), "a")
        logger_file.write("train completed, best dice: {} ".format(dice_val_best) +
            "per class: {} ".format(dice_val_list_best) +
            "at iteration: {}\n".format(global_step_best))
        logger_file.close()

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
        plt.savefig(os.path.join(root_dir, model_save_prefix + "_train_val.png"))

    # Example visualization
    model.load_state_dict(torch.load(os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")))
    model.eval()
    final_dice = np.load(os.path.join(root_dir, model_save_prefix + "_dice_values_list.npy"))
    final_precision = np.load(os.path.join(root_dir, model_save_prefix + "_precision_values.npy"))
    final_recall = np.load(os.path.join(root_dir, model_save_prefix + "_recall_values.npy"))
    final_hsd = np.load(os.path.join(root_dir, model_save_prefix + "_hsd_values.npy"))
    print(
        "best average dice and per class: {} ".format(final_dice) +
        "best average precision and per class: {} ".format(final_precision) +
        "best average recall and per class: {} ".format(final_recall) +
        "best average hsd and per class: {} ".format(final_hsd)
    )
    # Visualize
    with torch.no_grad():
        for case_num in range(len(val_ds)):
            val_inputs = val_ds[case_num]["image"].to(device)
            img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
            val_outputs = sliding_window_inference(val_inputs.unsqueeze(0), (crop_size, crop_size, crop_size),
                                                   4, predictor=model, overlap=0.8)
            val_output = post_pred(val_outputs[0]).cpu()
            val_label = val_ds[case_num]["label"]
            for slice_num in range(val_inputs.shape[-1]):
                unique_classes, counts = np.unique(val_label[0, :, :, slice_num].numpy(), return_counts=True)
                if len(unique_classes) < n_classes:
                    continue
                class_ratio = counts[1] / counts[0]
                # Plot only if the ground-truth is large enough
                if class_ratio < 0.05:
                    continue
                unique_classes, counts = np.unique(val_output[0, :, :, slice_num].numpy(), return_counts=True)
                if len(unique_classes) < n_classes:
                    continue
                class_ratio = counts[1] / counts[0]
                # Plot only if the ground-truth is large enough
                if class_ratio < 0.05:
                    continue
                plt.figure("example_{}_{}".format(img_name, slice_num), (18, 6))
                plt.subplot(1, 2, 1)
                plt.title("label")
                plt.imshow(val_inputs[0, :, :, slice_num].detach().cpu(), 'gray', interpolation='none')
                plt.imshow(val_label[0, :, :, slice_num], 'magma', interpolation='none', alpha=0.5)
                plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                plt.subplot(1, 2, 2)
                plt.title("prediction")
                plt.imshow(val_inputs[0, :, :, slice_num].detach().cpu(), 'gray', interpolation='none')
                # to make the segment colored, negate sign
                plt.imshow(1 - val_output[0, :, :, slice_num], 'magma', interpolation='none', alpha=0.5)
                plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                plt.savefig(os.path.join(root_dir, model_save_prefix + "_example_{}_{}.pdf".format(img_name, slice_num)))
                break