# --------------------------------------------------------
# This code is heavily borrowed from
# "https://github.com/valeoai/ADVENT/blob/master/advent/domain_adaptation/train_UDA.py"
# --------------------------------------------------------
import os
import sys
from pathlib import Path
from torch.autograd import Variable
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from sklearn import preprocessing

from advent.model.discriminator import get_fc_discriminator, get_fe_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.loss import WeightedBCEWithLogitsLoss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask


def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    # d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux = get_fe_discriminator(num_classes=1024)
    # saved_state_dict_D1 = torch.load('C:\\Users\\Administrator\\OneDrive - University of Ottawa\\Python\\ADVENT-master\experiments\\snapshots\\GTA2Cityscapes_DeepLabv2_AdvEnt413\\model_125000_D_aux.pth')
    # d_aux.load_state_dict(saved_state_dict_D1)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    # saved_state_dict_D2 = torch.load('C:\\Users\\Administrator\\OneDrive - University of Ottawa\\Python\\ADVENT-master\\experiments\\snapshots\\GTA2Cityscapes_DeepLabv2_AdvEnt413\\model_125000_D_main.pth')
    # d_main.load_state_dict(saved_state_dict_D2)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # interp_aux = nn.Upsample(size=(128, 256), mode='bilinear', align_corners=True)   # H/4
    # interp_aux_source = nn.Upsample(size=(180, 320), mode='bilinear', align_corners=True)   # H/4

    weighted_bce_loss = WeightedBCEWithLogitsLoss()
    criterion_seg = nn.CrossEntropyLoss(ignore_index=255)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    Epsilon = 0.1
    Lambda_local = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        damping = (1 - i_iter/100000)

        ### UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device)) # H/8 multi-level outputs coming from both conv4 and conv5
        # pred_src_aux = interp_aux_source(pred_src_aux)  # H/4=1280/4
        loss_seg_src_aux = 0
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_src_aux = interp(pred_src_aux)
        #     loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        #     # pred_src_aux = F.softmax(pred_src_aux1)
        # else:
        #     loss_seg_src_aux = 0

        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))  # H/8=120, H/8=129
        # pred_trg_aux = interp_aux(pred_trg_aux)  # H/4=256
        pred_trg_main_0 = interp_target(pred_trg_main)
        pred_trg_main = F.softmax(pred_trg_main_0)

        def toweight(x):

            x = x.cpu().data[0][0]
            x = preprocessing.scale(x)
            x = 1 / (1 + np.exp(-x))
            x = x * 1.5
            x = torch.tensor(x, dtype=torch.float32, device=device)

            return x

        if cfg.TRAIN.MULTI_LEVEL:
            # pred_trg_aux = F.softmax(interp_target(pred_trg_aux))
            # d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux))) # -p*log(p)
            d_out_aux = interp_target(d_aux(pred_trg_aux))  # H/8->H/8->H
            loss_adv_trg_aux = 0
            ones = torch.ones_like(d_out_aux)
            zero = torch.zeros_like(d_out_aux)

            # if (i_iter > 5000):
            #     pred_trg_aux_conf = 1.0 - torch.max(pred_trg_aux, 1)[0]
            #     weight_map_aux = torch.unsqueeze(pred_trg_aux_conf, dim=0)
            #     loss_adv_trg_aux = weighted_bce_loss(d_out_aux, Variable(torch.FloatTensor(d_out_aux.data.size()).fill_(source_label).to(device)),
            #                                          weight_map_aux, Epsilon , Lambda_local)
            # else:
            #     loss_adv_trg_aux = bce_loss(d_out_aux, source_label)

        else:
            loss_adv_trg_aux = 0

        # pred_trg_main = F.softmax(interp_target(pred_trg_main))  # H/8->H
        d_out_main = interp_target(d_main(pred_trg_main))  # H->H/8->H
        # loss_adv_trg_main = bce_loss(d_out_main, source_label)

        if (i_iter > 5000):

            maxpred, label = torch.max(pred_trg_main.detach(), dim=1)
            mask = (maxpred > 0.90)
            label = torch.where(mask, label, torch.ones(1).to(device, dtype=torch.long) * 255)
            loss_seg_trg_main = criterion_seg(pred_trg_main_0, label)
            # loss_seg_trg_main_.backward()

            pred_trg_main_conf = 1.0 - torch.max(pred_trg_main, 1)[0]
            fweight = toweight(d_out_aux)
            # pred_trg_main_conf = 1 - torch.max(pred_trg_main.detach(), 1)[0]
            # fweight = toweight(d_out_aux.detach())
            weight_map_main = pred_trg_main_conf * fweight
            weight_map_main = torch.where(weight_map_main > 1, ones, weight_map_main)
            weight_map_main = torch.where(weight_map_main < 0.05, zero, weight_map_main)

            # weight_map_main = torch.unsqueeze(weight_map_main, dim=0)
            loss_adv_trg_main = weighted_bce_loss(d_out_main,
                Variable(torch.FloatTensor(d_out_main.data.size()).fill_(source_label).to(device)),
                weight_map_main, Epsilon, Lambda_local)
        else:
            loss_adv_trg_main = bce_loss(d_out_main, source_label)
            loss_seg_trg_main = 0

        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main * damping
        loss.backward()

        ### Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            # d_out_aux = interp(d_aux(F.softmax(pred_src_aux))) # -plog(p)
            d_out_aux = interp(d_aux(pred_src_aux))  # H/8->H/8->H
            # d_out_aux = d_aux(prob_2_entropy(pred_src_aux))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = interp(d_main(F.softmax(pred_src_main)))  # H->H/8->H
        # d_out_main = d_main(prob_2_entropy(pred_src_main))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            # d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            d_out_aux = interp_target(d_aux(pred_trg_aux))  # H/8->H/8->H
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
            # if (i_iter > 5000):
            #     weight_map_aux = weight_map_aux.detach()
            #     loss_d_aux = weighted_bce_loss(d_out_aux, Variable(torch.FloatTensor(d_out_aux.data.size()).fill_(target_label).to(device)),
            #                                          weight_map_aux, Epsilon, Lambda_local)
            # else:
            #     loss_d_aux = bce_loss(d_out_aux, target_label)


        else:
            loss_d_aux = 0

        pred_trg_main = pred_trg_main.detach()
        d_out_main = interp_target(d_main(pred_trg_main))
        # loss_d_main = bce_loss(d_out_main, target_label)

        if (i_iter > 5000):
            pred_trg_main_conf = pred_trg_main_conf.detach()
            fweight = toweight(d_out_aux)
            # fweight = toweight(d_out_aux.detach())
            weight_map_main = pred_trg_main_conf * fweight
            weight_map_main = torch.where(weight_map_main > 1, ones, weight_map_main)
            # weight_map_main = torch.unsqueeze(weight_map_main, dim=0)
            loss_d_main = weighted_bce_loss(d_out_main, Variable(
                torch.FloatTensor(d_out_main.data.size()).fill_(target_label).to(device)),
                                            weight_map_main, Epsilon, Lambda_local)
        else:
            loss_d_main = bce_loss(d_out_main, target_label)

        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_trg_main': loss_seg_trg_main,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
