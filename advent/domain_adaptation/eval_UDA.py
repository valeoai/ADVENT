# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load


def evaluate_domain_adaptation(models, test_loader, cfg,
                               verbose=True):
    device = cfg.GPU_ID
    interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                         mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        name_classes = np.array(test_loader.dataset.info['label'], dtype=np.str)
        display_stats(cfg, name_classes, inters_over_union_classes)


def eval_best(cfg, models,
              device, test_loader, interp,
              verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        if verbose:
            name_classes = np.array(test_loader.dataset.info['label'], dtype=np.str)
            display_stats(cfg, name_classes, inters_over_union_classes)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
