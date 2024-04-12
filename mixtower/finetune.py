# ------------------------------------------------------------------------------ #
# Description: Runner that handles the finetuning and evaluation process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())

from datetime import datetime
import pickle, random, math, time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
from pathlib import Path
from copy import deepcopy
import yaml

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.mcan_for_finetune import MCANForFinetuneok, MCANForFinetunecoco
from .utils.optim import get_optim_for_finetune as get_optim
from .utils.data_parallel import BalancedDataParallel


def load_checkpoint(model, url_or_filename):
    if os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['state_dict']
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    for n, p in model.named_parameters():
        if n not in state_dict:
            state_dict[n] = p

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg


def FreezeBert(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'bert.' in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError


def FreezeVisual(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'sa_v.' in n:
            p.requires_grad = False
        if 'ffn_v.' in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError


class Runner(object):
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        
    def train(self, train_set, eval_set=None):
        data_size = train_set.data_size

        # Define the MCAN model
        if (self.__C.TASK == 'ok'):
            net = MCANForFinetuneok(self.__C, train_set.ans_size)
        elif (self.__C.TASK == 'coco'):
            net = MCANForFinetunecoco(self.__C, train_set.ans_size)
        else:
            print("model error....")

        # load the pretrained model
        if self.__C.PRETRAINED_MODEL_PATH is not None:
            net, msg = load_checkpoint(net, self.__C.PRETRAINED_MODEL_PATH)
            net.parameter_init()
            print('Finish loading.')

        # Define the optimizer
        if self.__C.RESUME:
            raise NotImplementedError('Resume training is not needed as the finetuning is fast')
        else:
            optim = get_optim(self.__C, net)
            start_epoch = 0

        # load to gpu
        net.cuda()
        if self.__C.N_GPU > 1:
            net = BalancedDataParallel(self.__C.GPU0_BS, net, device_ids=self.__C.GPU_IDS)

        # freeze bert
        if self.__C.FRZ_BERT:
            FreezeBert(net)

        # freeze visual
        if self.__C.FRZ_VIS:
            FreezeVisual(net)

        # Define the binary cross entropy loss
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        epoch_loss = 0

        # Define multi-thread dataloader
        dataloader = Data.DataLoader(train_set, batch_size=self.__C.BATCH_SIZE, shuffle=True, num_workers=self.__C.NUM_WORKERS, pin_memory=self.__C.PIN_MEM, drop_last=True)

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            net.train()
            # Save log information
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
            time_start = time.time()

            # Iteration
            for step, input_tuple in enumerate(dataloader):
                iteration_loss = 0
                optim.zero_grad()
                input_tuple = [x.cuda() for x in input_tuple]
                SUB_BATCH_SIZE = self.__C.BATCH_SIZE // self.__C.GRAD_ACCU_STEPS
                for accu_step in range(self.__C.GRAD_ACCU_STEPS):
                    sub_tuple = [x[accu_step * SUB_BATCH_SIZE: (accu_step + 1) * SUB_BATCH_SIZE] for x in input_tuple]
                    sub_ans_iter = sub_tuple[-1]
                    pred = net(sub_tuple[:-1])
                    loss = loss_fn(pred, sub_ans_iter)
                    loss.backward()
                    loss_item = loss.item()
                    iteration_loss += loss_item
                    epoch_loss += loss_item

                print("\r[version %s][epoch %2d][step %4d/%4d][Task %s][Mode %s] loss: %.4f, lr: %.2e" %
                      (self.__C.VERSION, epoch + 1, step, int(data_size / self.__C.BATCH_SIZE), self.__C.TASK, self.__C.RUN_MODE, iteration_loss/self.__C.BATCH_SIZE, optim.current_lr(),), end=' ')
                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end - time_start)))

            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'epoch = {epoch + 1}  loss = {epoch_loss / data_size}\nlr = {optim.current_lr()}\n\n')
            optim.schedule_step(epoch)

            # Save checkpoint
            state = {'state_dict': net.state_dict() if self.__C.N_GPU == 1 else net.module.state_dict()}
            torch.save(state, f'{self.__C.CKPTS_DIR}/epoch{epoch + 1}.pkl')

            # Eval after every epoch
            if eval_set is not None:
                self.eval(eval_set, net, eval_now=True)
            epoch_loss = 0

    # Evaluation
    @torch.no_grad()
    def eval(self, dataset, net=None, eval_now=False):
        data_size = dataset.data_size
        
        if net is None:
            # Load parameters
            path = self.__C.CKPT_PATH
            print('Loading ckpt {}'.format(path))
            # Define the MCAN model
            if (self.__C.TASK == 'ok'):
                net = MCANForFinetuneok(self.__C, dataset.ans_size)
            elif (self.__C.TASK == 'coco'):
                net = MCANForFinetunecoco(self.__C, dataset.ans_size)
            else:
                print("model error ...")
            ckpt = torch.load(path, map_location='cpu')
            net.load_state_dict(ckpt['state_dict'], strict=False)
            net.cuda()
            if self.__C.N_GPU > 1:
                net = BalancedDataParallel(self.__C.GPU0_BS, net, device_ids=self.__C.GPU_IDS)
            print('Finish!')

        net.eval()
        dataloader = Data.DataLoader(dataset, batch_size=self.__C.EVAL_BATCH_SIZE, shuffle=False, num_workers=self.__C.NUM_WORKERS, pin_memory=True)

        qid_idx = 0
        self.evaluater.init()
        for step, input_tuple in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (step, int(data_size / self.__C.EVAL_BATCH_SIZE),), end=' ')
            input_tuple = [x.cuda() for x in input_tuple]
            pred = net(input_tuple[:-1])
            pred_np = pred.cpu().numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # collect answers for every batch
            for i in range(len(pred_argmax)):
                qid = dataset.qids[qid_idx]
                qid_idx += 1
                ans_id = int(pred_argmax[i])
                ans = dataset.ix_to_ans[ans_id]
                # log result to evaluater
                self.evaluater.add(qid, ans)
        
        print()
        self.evaluater.save(self.__C.RESULT_PATH)
        # evaluate if eval_now is True
        if eval_now:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)

    def run(self):
        # ## Set ckpts and log path
        # where checkpoints will be saved
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        # where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        # where eval results will be saved
        Path(self.__C.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')

        # build dataset entities        
        common_data = CommonData(self.__C)
        if self.__C.RUN_MODE == 'finetune':
            train_set = DataSet(self.__C, common_data, self.__C.TRAIN_SPLITS)
            valid_set = None
            if self.__C.EVAL_NOW:
                valid_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS)
            self.train(train_set, valid_set)
        elif self.__C.RUN_MODE == 'finetune_test':
            test_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS)
            self.eval(test_set, eval_now=self.__C.EVAL_NOW)
        else:
            raise ValueError('Invalid run mode')


def finetune_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help='run mode', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--resume', dest='RESUME', help='resume training', type=bool, default=False)
    parser.add_argument('--resume_version', dest='RESUME_VERSION', help='checkpoint version name', type=str, default='')
    parser.add_argument('--resume_epoch', dest='RESUME_EPOCH', help='checkpoint epoch', type=int, default=1)
    parser.add_argument('--resume_path', dest='RESUME_PATH', help='checkpoint path', type=str, default='')
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=None)
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=None)
    parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    finetune_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()
