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
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandSpatialCropSamplesd,
    CropForegroundd,
)

from monai.config import print_config
from unetr import UNETR

from monai.data import (
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch
#torch.autograd.set_detect_anomaly(True)
from torch.nn import CosineSimilarity
import argparse
from itertools import product, permutations
import time

"""
Differences from standard global contrastive loss:
1) Learning without a FC projection layer
2) Using only global similarity of corresponding slice partitions
3) Repeating global similarity for each of the 3 axes
"""

def extract_triplets_more_partitions(batch1, batch2, slice_dimension=2):
    """
    Two volumes, each with a pair of transforms on the volume
    :param batch1: (batchsize, channels, x, y, z)
    :param batch2: (batchsize, channels, x, y, z)
    same batch indices in the two batches are different transforms of the same volume
    assuming batch_size = 2
    """
    print("Shapes of loss batch pair", batch1.shape, batch2.shape)
    dims = batch1.shape
    # choose slicing dimension
    slices_list = []
    if slice_dimension == 2:
        print("Slicing dimension", slice_dimension)
        partition_size = int(dims[slice_dimension] / num_partitions)
        # for each sampled volume, we sample one image per partition
        init_idx = np.random.choice(np.arange(0, partition_size))
        slice_idx_list = [init_idx + partition_idx * partition_size for partition_idx in range(num_partitions)]
        print("Slice indices:", slice_idx_list)
        for slice_idx in slice_idx_list:
            # Two volumes, each with a pair of transforms on the volume
            x1_slice = batch1[0, :, slice_idx].reshape(dims[1], -1)  # channels x flattened features
            x1_trans_slice = batch1[1, :, slice_idx].reshape(dims[1], -1)
            x2_slice = batch2[0, :, slice_idx].reshape(dims[1], -1)
            x2_trans_slice = batch2[1, :, slice_idx].reshape(dims[1], -1)
            slices_list.append([x1_slice, x1_trans_slice, x2_slice, x2_trans_slice])
            print(x1_slice.shape, x1_trans_slice.shape, x2_slice.shape, x2_trans_slice.shape)
    if slice_dimension == 3:
        print("Slicing dimension", slice_dimension)
        partition_size = int(dims[slice_dimension] / num_partitions)
        # for each sampled volume, we sample one image per partition
        init_idx = np.random.choice(np.arange(0, partition_size))
        slice_idx_list = [init_idx + partition_idx * partition_size for partition_idx in range(num_partitions)]
        print("Slice indices:", slice_idx_list)
        for slice_idx in slice_idx_list:
            # Two volumes, each with a pair of transforms on the volume
            x1_slice = batch1[0, :, :, slice_idx].reshape(dims[1], -1)  # channels x flattened features
            x1_trans_slice = batch1[1, :, :, slice_idx].reshape(dims[1], -1)
            x2_slice = batch2[0, :, :, slice_idx].reshape(dims[1], -1)
            x2_trans_slice = batch2[1, :, :, slice_idx].reshape(dims[1], -1)
            slices_list.append([x1_slice, x1_trans_slice, x2_slice, x2_trans_slice])
            print(x1_slice.shape, x1_trans_slice.shape, x2_slice.shape, x2_trans_slice.shape)
    if slice_dimension == 4:
        print("Slicing dimension", slice_dimension)
        partition_size = int(dims[slice_dimension] / num_partitions)
        # for each sampled volume, we sample one image per partition
        init_idx = np.random.choice(np.arange(0, partition_size))
        slice_idx_list = [init_idx + partition_idx * partition_size for partition_idx in range(num_partitions)]
        print("Slice indices:", slice_idx_list)
        for slice_idx in slice_idx_list:
            # Two volumes, each with a pair of transforms on the volume
            x1_slice = batch1[0, :, :, :, slice_idx].reshape(dims[1], -1)  # channels x flattened features
            x1_trans_slice = batch1[1, :, :, :, slice_idx].reshape(dims[1], -1)
            x2_slice = batch2[0, :, :, :, slice_idx].reshape(dims[1], -1)
            x2_trans_slice = batch2[1, :, :, :, slice_idx].reshape(dims[1], -1)
            slices_list.append([x1_slice, x1_trans_slice, x2_slice, x2_trans_slice])
            print(x1_slice.shape, x1_trans_slice.shape, x2_slice.shape, x2_trans_slice.shape)
    reference = []
    similar = []
    dissimilar = []
    # for each partition
    for partition_idx in range(num_partitions):
        # same slice same volume different transforms are similar (from SimCLR)
        # same slice different volumes and their transformations are similar (from medical image segmentation)
        # different slices and their transformations are dissimilar
        current_partition = slices_list[partition_idx]
        other_partitions = []
        for other_partition_idx in range(num_partitions):
            if not (other_partition_idx == partition_idx):
                other_partitions.extend(slices_list[other_partition_idx])
        for (ref_sim_pair, dissim) in product(permutations(current_partition, 2), other_partitions):
            reference.append(ref_sim_pair[0])
            similar.append(ref_sim_pair[1])
            dissimilar.append(dissim)
    return reference, similar, dissimilar

def extract_triplets(batch1, batch2, slice_dimension=2):
    """
    Two volumes, each with a pair of transforms on the volume
    :param batch1: (batchsize, channels, x, y, z)
    :param batch2: (batchsize, channels, x, y, z)
    same batch indices in the two batches are different transforms of the same volume
    assuming batch_size = 2 and num_partitions = 2
    """
    print("Shapes of loss batch pair", batch1.shape, batch2.shape)
    dims = batch1.shape
    # choose slicing dimension
    if slice_dimension == 2:
        print("Slicing dimension", slice_dimension)
        slice1_idx = np.random.choice(np.arange(0, int(dims[slice_dimension] / 2)))
        slice2_idx = np.random.choice(np.arange(int(dims[slice_dimension] / 2), dims[slice_dimension]))
        x1_slice1 = batch1[0, :, slice1_idx].reshape(dims[1], -1)  # channels x flattened features
        x1_trans_slice1 = batch1[1, :, slice1_idx].reshape(dims[1], -1)
        x1_slice2 = batch1[0, :, slice2_idx].reshape(dims[1], -1)
        x1_trans_slice2 = batch1[1, :, slice2_idx].reshape(dims[1], -1)
        x2_slice1 = batch2[0, :, slice1_idx].reshape(dims[1], -1)
        x2_trans_slice1 = batch2[1, :, slice1_idx].reshape(dims[1], -1)
        x2_slice2 = batch2[0, :, slice2_idx].reshape(dims[1], -1)
        x2_trans_slice2 = batch2[1, :, slice2_idx].reshape(dims[1], -1)
    if slice_dimension == 3:
        print("Slicing dimension", slice_dimension)
        slice1_idx = np.random.choice(np.arange(0, int(dims[slice_dimension] / 2)))
        slice2_idx = np.random.choice(np.arange(int(dims[slice_dimension] / 2), dims[slice_dimension]))
        x1_slice1 = batch1[0, :, :, slice1_idx].reshape(dims[1], -1)
        x1_trans_slice1 = batch1[1, :, :, slice1_idx].reshape(dims[1], -1)
        x1_slice2 = batch1[0, :, :, slice2_idx].reshape(dims[1], -1)
        x1_trans_slice2 = batch1[1, :, :, slice2_idx].reshape(dims[1], -1)
        x2_slice1 = batch2[0, :, :, slice1_idx].reshape(dims[1], -1)
        x2_trans_slice1 = batch2[1, :, :, slice1_idx].reshape(dims[1], -1)
        x2_slice2 = batch2[0, :, :, slice2_idx].reshape(dims[1], -1)
        x2_trans_slice2 = batch2[1, :, :, slice2_idx].reshape(dims[1], -1)
    if slice_dimension == 4:
        print("Slicing dimension", slice_dimension)
        slice1_idx = np.random.choice(np.arange(0, int(dims[slice_dimension] / 2)))
        slice2_idx = np.random.choice(np.arange(int(dims[slice_dimension] / 2), dims[slice_dimension]))
        x1_slice1 = batch1[0, :, :, :, slice1_idx].reshape(dims[1], -1)
        x1_trans_slice1 = batch1[1, :, :, :, slice1_idx].reshape(dims[1], -1)
        x1_slice2 = batch1[0, :, :, :, slice2_idx].reshape(dims[1], -1)
        x1_trans_slice2 = batch1[1, :, :, :, slice2_idx].reshape(dims[1], -1)
        x2_slice1 = batch2[0, :, :, :, slice1_idx].reshape(dims[1], -1)
        x2_trans_slice1 = batch2[1, :, :, :, slice1_idx].reshape(dims[1], -1)
        x2_slice2 = batch2[0, :, :, :, slice2_idx].reshape(dims[1], -1)
        x2_trans_slice2 = batch2[1, :, :, :, slice2_idx].reshape(dims[1], -1)
    print(x1_slice1.shape, x1_slice2.shape, x2_slice1.shape, x2_slice2.shape)
    reference = []
    similar = []
    dissimilar = []
    # same slice same volume different transforms are similar (from SimCLR)
    # same slice different volumes and their transformations are similar (from medical image segmentation)
    # different slices and their transformations are dissimilar
    # add both directions of ranking 1->2 2->1
    for (ref_sim_pair, dissim) in product(permutations([x1_slice1, x1_trans_slice1, x2_slice1, x2_trans_slice1], 2),
                                      [x1_slice2, x1_trans_slice2, x2_slice2, x2_trans_slice2]):
        reference.append(ref_sim_pair[0])
        similar.append(ref_sim_pair[1])
        dissimilar.append(dissim)
    for (ref_sim_pair, dissim) in product(permutations([x1_slice2, x1_trans_slice2, x2_slice2, x2_trans_slice2], 2),
                                      [x1_slice1, x1_trans_slice1, x2_slice1, x2_trans_slice1]):
        reference.append(ref_sim_pair[0])
        similar.append(ref_sim_pair[1])
        dissimilar.append(dissim)
    return reference, similar, dissimilar

def BTLoss(reference, similar, dissimilar, optimizer):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    Focus on global contrastive
    """
    start_time = time.time()
    loss = 0
    for ref, sim, dissim in zip(reference, similar, dissimilar):
        similar_pred = cos(ref, sim) / temperature
        dissimilar_pred = cos(ref, dissim) / temperature
        comp_pred = similar_pred - dissimilar_pred  # si-sj
        loss += torch.mean(torch.log(1 + torch.exp((comp_pred))))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    cum_loss = loss.item()
    end_time = time.time()
    return cum_loss, end_time - start_time

def ContrastiveLoss(reference, similar, dissimilar, optimizer):
    """
    Contrastive learning of global and local features for
    medical image segmentation with limited annotations
    Focus on global contrastive
    """
    start_time = time.time()
    loss = 0
    for ref, sim in zip(reference, similar):
        numerator = torch.exp(cos(ref, sim) / temperature)
        denominator_list = [torch.exp(cos(ref, dissim) / temperature) for dissim in dissimilar]
        denominator_list.append(numerator)
        denominator = torch.stack(denominator_list, dim=0).sum(dim=0)
        loss += - torch.mean(torch.log(numerator / denominator))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    cum_loss = loss.item()
    end_time = time.time()
    return cum_loss, end_time - start_time

def train(global_step, train_loader, update_arc, model_save_prefix):
    model.train()
    # repeat global similarity learning for each of the 3 dimensions
    for slice_dimension in [2, 3, 4]:
        epoch_ranking_loss = 0
        epoch_time = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (ranking loss=X.X) (loss time=X.X)", dynamic_ncols=True
        )
        # load 2 transforms of the same volume
        for step, batch in enumerate(epoch_iterator):
            # batch is concatenation of Two volumes, each with a pair of transforms on the volume
            print("Input batch shape", batch["image"].shape)
            step += 1
            # concat two different transforms
            x = batch["image"].cuda()
            # fw pass
            if update_arc == "feat":
                input, _ = model(x)  # latent features are the input
            else:
                _, input = model(x, freeze_encoder=True)  # decoder output is the input
            # split features into two different volumes
            f1, f2 = torch.split(input, [batch_size, batch_size], dim=0)
            # create triplets
            if update_arc == "feat":
                reference, similar, dissimilar = extract_triplets(f1, f2, slice_dimension)
            else:
                reference, similar, dissimilar = extract_triplets_more_partitions(f1, f2, slice_dimension)
            # loss and optimize
            if args.loss == "ranking":
                ranking_loss, loss_time = BTLoss(reference, similar, dissimilar, optimizer)
            else:
                ranking_loss, loss_time = ContrastiveLoss(reference, similar, dissimilar, optimizer)
            epoch_ranking_loss += ranking_loss
            epoch_time += loss_time
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (ranking loss=%2.5f) (loss time=%2.5f)" %
                (global_step, max_iterations, ranking_loss, loss_time)
            )

            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                #epoch_ranking_loss /= step
                #epoch_time /= step
                epoch_ranking_loss_values.append(epoch_ranking_loss)
                epoch_time_values.append(epoch_time)
                torch.save(
                    model.state_dict(), os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")
                )
                print(
                    "Model Was Saved At Global Step {} for {}!".format(global_step, update_arc)
                )
            global_step += 1
    return global_step


if __name__ == '__main__':
    """
    python unetr_btcv_ranking_pretraining_3d.py "./dataset/" "./results" 14 50 16 0.0001 0.1 "ranking"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default="./dataset/")
    parser.add_argument('root_dir', type=str, default="./results")
    parser.add_argument('n_classes', type=int, default=14)
    parser.add_argument('train_size', type=int, default=50)
    parser.add_argument('crop_size', type=int, default=16)
    parser.add_argument('learning_rate', type=float, default=0.0001)
    parser.add_argument('temperature', type=float, default=0.1)
    parser.add_argument('loss', type=str, default="ranking")
    args = parser.parse_args()

    # Dataset parameters
    n_classes = args.n_classes
    train_size = args.train_size
    data_dir = args.data_dir

    # Tuned parameters based on dataset
    root_dir = args.root_dir + "_" + args.loss
    learning_rate = args.learning_rate
    temperature = args.temperature
    crop_size = args.crop_size  # bottleneck features are 2 dimensional

    # fixed for now
    num_partitions = 4
    batch_size = 2

    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

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
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=(crop_size, crop_size, crop_size), random_size=False,
                num_samples=batch_size,  # a pair of transforms on each volume
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

    # Data loader
    split_JSON = "dataset_0.json"
    datasets = data_dir + split_JSON
    datalist = load_decathlon_datalist(datasets, True, "training")
    train_ds = Dataset(
        data=datalist,
        transform=train_transforms,  # Create two transforms of the same volume
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Architecture
    model = UNETR(
        in_channels=1,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    cos = CosineSimilarity(dim=-1, eps=1e-6)

    # Training
    max_iterations = 10000
    eval_num = 50
    rtol = 1e-2

    # update features
    params_conv = False
    global_step = 0
    epoch_ranking_loss_values = []
    epoch_time_values = []
    update_arc = "feat"
    model_save_prefix = "{}_lr_{}_temp_{}".format(update_arc, learning_rate, temperature)
    while not params_conv:
        global_step = train(global_step, train_loader, update_arc, model_save_prefix)
        if global_step <= eval_num:
            avg_obj = np.mean(epoch_ranking_loss_values[:-1])
        else:
            avg_obj = np.mean(epoch_ranking_loss_values[-eval_num - 1:-1])
        params_conv = np.abs(avg_obj - epoch_ranking_loss_values[-1]) < rtol * avg_obj or \
                      global_step >= max_iterations  # check conv.
    print(
        "Training Converged At Global Step {} for {}!".format(global_step, update_arc)
    )

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")))
    plt.figure("train", (12, 6))
    plt.title("Iteration Average Loss")
    x = [np.sum(epoch_time_values[:i+1]) for i in range(len(epoch_time_values))]
    y = epoch_ranking_loss_values
    plt.xlabel("Time(s)")
    plt.plot(x, y)
    plt.savefig(os.path.join(root_dir, model_save_prefix + "_train.png"))
    plt.close()

    # update reconstructions, freezing encoder
    params_conv = False
    global_step = 0
    epoch_ranking_loss_values = []
    epoch_time_values = []
    update_arc = "recon"
    model_save_prefix = "{}_lr_{}_temp_{}".format(update_arc, learning_rate, temperature)
    while not params_conv:
        global_step = train(global_step, train_loader, update_arc, model_save_prefix)
        if global_step <= eval_num:
            avg_obj = np.mean(epoch_ranking_loss_values[:-1])
        else:
            avg_obj = np.mean(epoch_ranking_loss_values[-eval_num - 1:-1])
        params_conv = np.abs(avg_obj - epoch_ranking_loss_values[-1]) < rtol * avg_obj or \
                      global_step >= max_iterations  # check conv.
    print(
        "Training Converged At Global Step {} for {}!".format(global_step, update_arc)
    )

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(root_dir, model_save_prefix + "_best_metric_model.pth")))
    plt.figure("train", (12, 6))
    plt.title("Iteration Average Loss")
    x = [np.sum(epoch_time_values[:i+1]) for i in range(len(epoch_time_values))]
    y = epoch_ranking_loss_values
    plt.xlabel("Time(s)")
    plt.plot(x, y)
    plt.savefig(os.path.join(root_dir, model_save_prefix + "_train.png"))