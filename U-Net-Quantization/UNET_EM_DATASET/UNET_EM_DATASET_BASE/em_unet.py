import torch
#from model import *
from data import *
import models as m
import config
from datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import defaultdict
import time
import os
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from transforms import *
from losses import *
from metrics import *
from filters import *
import utility
import config
import layover as lay
import natsort
from matplotlib.image import imread
import torch.nn.functional as F
import model_visualizer as mv
cudnn.benchmark = True


# # Force pytorch to only use cpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cuda = torch.cuda.is_available()
device = []


def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def get_samples(image_val_path, target_val_path, num_samples=4):
    # import ipdb as pdb; pdb.set_trace()
    image_val = torch.ones([4, 1, 200, 200])
    target_val = torch.ones([4, 1, 200, 200])
    for i in range(0, num_samples):
        image_val[i, 0, :, :] = torch.from_numpy(imread(image_val_path[i]))
        target_val[i, 0, :, :] = torch.from_numpy(imread(target_val_path[i]))
    return image_val, target_val


def save_pred(model, image_val_path, epoch, path):
    #import ipdb as pdb; pdb.set_trace()
    model.eval()
    image_val = torch.ones([4, 1, 200, 200])
    for i in range(0, 4):
        image_val[i, 0, :, :] = torch.from_numpy(imread(image_val_path[i]))
    preds = model(image_val.to(device))
    preds = preds.data.cpu().numpy()
    save_image(image_val[0][0], preds[0][0], epoch, 0, path)
    model.train()


def save_image(in_img, gt_img, pred_img, epoch, idx, path):
    # if epoch % 5 == 0 :
    in_img *= 255
    pred_img *= 255
    gt_img *= 255
    pred_img = pred_img.astype(int)
    # import ipdb as pdb; pdb.set_trace()
    # if epoch > 198:
    #     import ipdb as pdb; pdb.set_trace()
    img_name = "EM_UNET_" + str(idx) + str(epoch) + ".png"
    orig_img_name = "EM_UNET_ORIG_" + str(idx) + str(epoch) + ".png"
    gt_img_name = "EM_UNET_GT_" + str(idx) + str(epoch) + ".png"
    pred_img_name = "EM_UNET_PRED_" + str(idx) + str(epoch) + ".png"
    utility.save_image_to_path(in_img, orig_img_name, path)
    utility.save_image_to_path(gt_img, gt_img_name, path)
    utility.save_image_to_path(pred_img, pred_img_name, path)

    gt_over_layed = lay.lay_over(
        gt_img,   in_img,        lay.yellow, alpha=0.2, mask_val=255)
    pred_over_layed = lay.lay_over(
        pred_img, gt_over_layed, lay.red,    alpha=0.2, mask_val=255)
    utility.save_image_to_path(pred_over_layed, img_name, path)


def run_main(config):
    train_transform = transforms.Compose([
        CenterCrop2D((200, 200)),
        ElasticTransform(alpha_range=(28.0, 30.0),
                         sigma_range=(3.5, 4.0),
                         p=0.3),
        RandomAffine(degrees=4.6,
                     scale=(0.98, 1.02),
                     translate=(0.03, 0.03)),
        RandomTensorChannelShift((-0.10, 0.10)),
        ToTensor(),
        NormalizeInstance(),
    ])

    val_transform = transforms.Compose([
        CenterCrop2D((200, 200)),
        ToTensor(),
        NormalizeInstance(),
    ])
    # import ipdb as pdb; pdb.set_trace()
    dataset_base_path = "export/nana/datasets/em_challenge/"
    target_path = natsort.natsorted(
        glob.glob(dataset_base_path + 'mask/*.PNG'))
    image_paths = natsort.natsorted(
        glob.glob(dataset_base_path + 'data/*.PNG'))
    target_val_path = natsort.natsorted(
        glob.glob(dataset_base_path + 'val_mask/*.PNG'))
    image_val_path = natsort.natsorted(
        glob.glob(dataset_base_path + 'val_img/*.PNG'))

    gmdataset_train = EMdataset(
        image_paths=image_paths, target_paths=target_path)
    gmdataset_val = EMdataset(
        image_paths=image_val_path, target_paths=target_val_path)
    train_loader = DataLoader(gmdataset_train, batch_size=5,
                              shuffle=True,
                              num_workers=1)
    val_loader = DataLoader(gmdataset_val, batch_size=4,
                            shuffle=True,
                            num_workers=1)

    utility.create_log_file(config)
    utility.log_info(config, "{0}\nStarting experiment {1}\n{0}\n".format(
        50*"=", utility.get_experiment_name(config)))
    # import ipdb as pdb; pdb.set_trace()
    model = m.Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    if config['operation_mode'].lower() == "retrain" or config['operation_mode'].lower() == "inference":
        print("Using a trained model...")
        model.load_state_dict(torch.load(
            config['trained_model']))
    elif config["operation_mode"].lower() == "visualize":
        print("Using a trained model...")
        if cuda:
            model.load_state_dict(torch.load(config['trained_model']))
        else:
            model.load_state_dict(torch.load(
                config['trained_model'], map_location='cpu'))
        mv.visualize_model(model, config)
        return

    # import ipdb as pdb; pdb.set_trace()
    if cuda:
        model.cuda()

    num_epochs = config["num_epochs"]
    initial_lr = config["lr"]
    experiment_path = config["log_output_dir"] + config['experiment_name']
    output_image_dir = experiment_path + "/figs/"

    betas = torch.linspace(3.0, 8.0, num_epochs)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    lr_milestones = range(0, int(num_epochs), int(int(num_epochs)/5))
    lr_milestones = lr_milestones[1:]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=0.1)

    # import ipdb as pdb; pdb.set_trace()
    writer = SummaryWriter(log_dir=utility.get_experiment_dir(config))
    best_dice = 0
    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()

        scheduler.step()

        lr = scheduler.get_lr()[0]
        model.beta = betas[epoch-1]  # for ternary net, set beta
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        capture = True
        for i, batch in enumerate(train_loader):
            #import ipdb as pdb; pdb.set_trace()
            input_samples, gt_samples, idx = batch[0], batch[1], batch[2]

            if cuda:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(non_blocking=True)
                var_gt = var_gt.float()
            else:
                var_input = input_samples
                var_gt = gt_samples
                var_gt = var_gt.float()
            preds = model(var_input)

            # import ipdb as pdb; pdb.set_trace()
            loss = calc_loss(preds, var_gt)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1
            if epoch % 5 == 0 and capture:
                capture = False
                input_samples, gt_samples = get_samples(
                    image_val_path, target_val_path, 4)
                if cuda:
                    input_samples = input_samples.cuda()
                preds = model(input_samples)
                input_samples = input_samples.data.cpu().numpy()
                preds = preds.data.cpu().numpy()
                # import ipdb as pdb; pdb.set_trace()
                save_image(input_samples[0][0], gt_samples[0]
                           [0], preds[0][0], epoch, 0, output_image_dir)

        train_loss_total_avg = train_loss_total / num_steps

        # import ipdb as pdb; pdb.set_trace()
        model.train()
        val_loss_total = 0.0
        num_steps = 0

        metric_fns = [dice_score,
                      hausdorff_score,
                      precision_score,
                      recall_score,
                      specificity_score,
                      intersection_over_union,
                      accuracy_score,
                      rand_index_score]

        metric_mgr = MetricManager(metric_fns)

        for i, batch in enumerate(val_loader):
            #            input_samples, gt_samples = batch[0], batch[1]
            input_samples, gt_samples, idx = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda(non_blocking=True)
                    var_gt = var_gt.float()
                else:
                    var_input = input_samples
                    var_gt = gt_samples
                    var_gt = var_gt.float()
                # import ipdb as pdb; pdb.set_trace()

                start = time.time()
                preds = model(var_input)
                end = time.time()

                print("Inference Time : ", end-start, "----"*5)

                loss = dice_loss(preds, var_gt)
                val_loss_total += loss.item()
            # Metrics computation
            gt_npy = gt_samples.numpy().astype(np.uint8)
            gt_npy = gt_npy.squeeze(axis=1)

            preds = preds.data.cpu().numpy()
            preds = threshold_predictions(preds)
            preds = preds.astype(np.uint8)
            preds = preds.squeeze(axis=1)
            metric_mgr(preds, gt_npy)
            # save_image(input_samples[0][0], preds[0],
            #            gt_samples, epoch, idx[0], output_image_dir)
            # save_pred(model, image_val_path, epoch, output_image_dir)
            num_steps += 1

        metrics_dict = metric_mgr.get_results()
        metric_mgr.reset()

        writer.add_scalars('metrics', metrics_dict, epoch)

        # import ipdb as pdb; pdb.set_trace()
        val_loss_total_avg = val_loss_total / num_steps

        writer.add_scalars('losses', {
            'val_loss': val_loss_total_avg,
            'train_loss': train_loss_total_avg
        }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        # import ipdb as pdb; pdb.set_trace()
        log_str = "Epoch {} took {:.2f} seconds train_loss={}   dice_score={}   rand_index_score={}  lr={}.".format(
            epoch, total_time, train_loss_total_avg, metrics_dict["dice_score"], metrics_dict["rand_index_score"], get_lr(optimizer))
        utility.log_info(config, log_str)
        tqdm.write(log_str)

        writer.add_scalars('losses', {
            'train_loss': train_loss_total_avg
        }, epoch)
        if metrics_dict["dice_score"] > best_dice:
            best_dice = metrics_dict["dice_score"]
            utility.save_model(model=model, config=config)
    if not (config['operation_mode'].lower() == "inference"):
        utility.save_model(model=model, config=config)


if __name__ == '__main__':
    args = utility.parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    # import ipdb as pdb; pdb.set_trace()
    if cuda:
        if "gpu_core_num" in config:
            device_id = int(config["gpu_core_num"])
            torch.cuda.set_device(0)
            device = torch.device("cuda:{0}".format(
                device_id) if torch.cuda.is_available() else "cpu")
    run_main(config)
