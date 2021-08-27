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
from itertools import product, combinations


def extract_triplets(batch1, batch2):
    """
    Two different transforms of the same volumes
    :param batch1: (batchsize, channels, x, y, z)
    :param batch2: (batchsize, channels, x, y, z)
    same batch indices in the two batches are different transforms of the same volume
    assuming batch_size = 2 and num_partitions = 2
    """
    print("Shapes of loss batch pair", batch1.shape, batch2.shape)
    dims = batch1.shape
    # choose slicing dimension
    slice_dimension = np.random.choice([2, 3, 4])
    if slice_dimension == 2:
        print("Slicing dimension", slice_dimension)
        slice1_idx = np.random.choice(np.arange(0, int(dims[slice_dimension] / num_partitions)))
        slice2_idx = np.random.choice(np.arange(int(dims[slice_dimension] / num_partitions), dims[slice_dimension]))
        x1_slice1 = batch1[0, :, slice1_idx].reshape(dims[1], -1)  # channels x flattened features
        x1_trans_slice1 = batch2[0, :, slice1_idx].reshape(dims[1], -1)
        x1_slice2 = batch1[0, :, slice2_idx].reshape(dims[1], -1)
        x1_trans_slice2 = batch2[0, :, slice2_idx].reshape(dims[1], -1)
        x2_slice1 = batch1[1, :, slice1_idx].reshape(dims[1], -1)
        x2_trans_slice1 = batch2[1, :, slice1_idx].reshape(dims[1], -1)
        x2_slice2 = batch1[1, :, slice2_idx].reshape(dims[1], -1)
        x2_trans_slice2 = batch2[1, :, slice2_idx].reshape(dims[1], -1)
    if slice_dimension == 3:
        print("Slicing dimension", slice_dimension)
        slice1_idx = np.random.choice(np.arange(0, int(dims[slice_dimension] / num_partitions)))
        slice2_idx = np.random.choice(np.arange(int(dims[slice_dimension] / num_partitions), dims[slice_dimension]))
        x1_slice1 = batch1[0, :, :, slice1_idx].reshape(dims[1], -1)
        x1_trans_slice1 = batch2[0, :, :, slice1_idx].reshape(dims[1], -1)
        x1_slice2 = batch1[0, :, :, slice2_idx].reshape(dims[1], -1)
        x1_trans_slice2 = batch2[0, :, :, slice2_idx].reshape(dims[1], -1)
        x2_slice1 = batch1[1, :, :, slice1_idx].reshape(dims[1], -1)
        x2_trans_slice1 = batch2[1, :, :, slice1_idx].reshape(dims[1], -1)
        x2_slice2 = batch1[1, :, :, slice2_idx].reshape(dims[1], -1)
        x2_trans_slice2 = batch2[1, :, :, slice2_idx].reshape(dims[1], -1)
    if slice_dimension == 4:
        print("Slicing dimension", slice_dimension)
        slice1_idx = np.random.choice(np.arange(0, int(dims[slice_dimension] / num_partitions)))
        slice2_idx = np.random.choice(np.arange(int(dims[slice_dimension] / num_partitions), dims[slice_dimension]))
        x1_slice1 = batch1[0, :, :, :, slice1_idx].reshape(dims[1], -1)
        x1_trans_slice1 = batch2[0, :, :, :, slice1_idx].reshape(dims[1], -1)
        x1_slice2 = batch1[0, :, :, :, slice2_idx].reshape(dims[1], -1)
        x1_trans_slice2 = batch2[0, :, :, :, slice2_idx].reshape(dims[1], -1)
        x2_slice1 = batch1[1, :, :, :, slice1_idx].reshape(dims[1], -1)
        x2_trans_slice1 = batch2[1, :, :, :, slice1_idx].reshape(dims[1], -1)
        x2_slice2 = batch1[1, :, :, :, slice2_idx].reshape(dims[1], -1)
        x2_trans_slice2 = batch2[1, :, :, :, slice2_idx].reshape(dims[1], -1)
    print(x1_slice1.shape, x1_slice2.shape, x2_slice1.shape, x2_slice2.shape)
    reference = []
    similar = []
    dissimilar = []
    # same slice different transforms (from SimCLR)
    # different slices (from medical image segmentation)
    # dont make same partitions across different volumes dissimilar
    for (ref_sim_pair, dissim) in product(combinations([x1_slice1, x1_trans_slice1, x2_slice1, x2_trans_slice1], 2),
                                      [x1_slice2, x1_trans_slice2, x2_slice2, x2_trans_slice2]):
        reference.append(ref_sim_pair[0])
        similar.append(ref_sim_pair[1])
        dissimilar.append(dissim)
    for (ref_sim_pair, dissim) in product(combinations([x1_slice2, x1_trans_slice2, x2_slice2, x2_trans_slice2], 2),
                                      [x1_slice1, x1_trans_slice1, x2_slice1, x2_trans_slice1]):
        reference.append(ref_sim_pair[0])
        similar.append(ref_sim_pair[1])
        dissimilar.append(dissim)
    return reference, similar, dissimilar

def BTLoss(reference, similar, dissimilar, optimizer):
    """
    Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
    """
    cum_loss = 0
    for ref, sim, dissim in zip(reference, similar, dissimilar):
        similar_pred = cos(ref, sim) / temperature
        dissimilar_pred = cos(ref, dissim) / temperature
        comp_pred = similar_pred - dissimilar_pred  # si-sj
        #print("Comparison dimension", comp_pred.shape)
        loss = torch.mean(torch.log(1 + torch.exp((comp_pred))))
        loss.backward(retain_graph=True)
        cum_loss += loss.item()
    optimizer.step()
    optimizer.zero_grad()
    return cum_loss

def ContrastiveLoss(reference, similar, dissimilar, optimizer):
    """
    Contrastive learning of global and local features for
    medical image segmentation with limited annotations
    """
    cum_loss = 0
    for ref, sim in zip(reference, similar):
        numerator = torch.exp(cos(ref, sim) / temperature)
        denominator = numerator.clone()
        for dissim in dissimilar:
            denominator += torch.exp(cos(ref, dissim) / temperature)
        loss = - torch.mean(torch.log(numerator / denominator))
        loss.backward(retain_graph=True)
        cum_loss += loss.item()
    optimizer.step()
    optimizer.zero_grad()
    return cum_loss

def train(global_step, train_loader, update_arc="feat"):
    model.train()
    epoch_ranking_loss = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    # load 2 transforms of the same volume
    for step, batch in enumerate(epoch_iterator):
        # batch is concatenation of two volumes with 2 transformations on each
        print("Input batch shape", batch["image"].shape)
        step += 1
        # concat two different transforms
        x = batch["image"].cuda()
        # fw pass
        latent_feat, logit_map = model(x)
        if update_arc == "feat":
            input = latent_feat.clone()
        else:
            input = logit_map.clone()
        # split features
        f1, f2 = torch.split(input, [batch_size, batch_size], dim=0)
        # create triplets
        reference, similar, dissimilar = extract_triplets(f1, f2)
        # loss and optimize
        if args.loss == "ranking":
            ranking_loss = BTLoss(reference, similar, dissimilar, optimizer)
        else:
            ranking_loss = ContrastiveLoss(reference, similar, dissimilar, optimizer)
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
                model.state_dict(), os.path.join(root_dir, update_arc + "_best_metric_model.pth")
            )
            print(
                "Model Was Saved At Global Step {} for {}!".format(global_step, update_arc)
            )
        global_step += 1
    return global_step


if __name__ == '__main__':
    """
    python unetr_btcv_ranking_pretraining_3d.py "./dataset/" "./results" 14 50 16 0.0001 0.5 "ranking"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default="./dataset/")
    parser.add_argument('root_dir', type=str, default="./results")
    parser.add_argument('n_classes', type=int, default=14)
    parser.add_argument('train_size', type=int, default=50)
    parser.add_argument('crop_size', type=int, default=16)
    parser.add_argument('learning_rate', type=float, default=0.0001)
    parser.add_argument('temperature', type=float, default=0.5)
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
    num_partitions = 2
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
                num_samples=batch_size,  # a pair of transforms on each crop
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
    max_iterations = 15000
    eval_num = 500

    # update features
    global_step = 0
    epoch_ranking_loss_values = []
    update_arc = "feat"
    while global_step < max_iterations:
        global_step = train(global_step, train_loader, update_arc)

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(root_dir, update_arc + "_best_metric_model.pth")))
    plt.figure("train", (12, 6))
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_ranking_loss_values))]
    y = epoch_ranking_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.savefig(os.path.join(root_dir, update_arc + "_train.png"))

    # update reconstructions
    global_step = 0
    epoch_ranking_loss_values = []
    update_arc = "recon"
    while global_step < max_iterations:
        global_step = train(global_step, train_loader, update_arc)

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(root_dir, update_arc + "_best_metric_model.pth")))
    plt.figure("train", (12, 6))
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_ranking_loss_values))]
    y = epoch_ranking_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.savefig(os.path.join(root_dir, update_arc + "_train.png"))