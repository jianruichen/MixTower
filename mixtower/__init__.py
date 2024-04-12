__author__ = 'Deguang Chen'
__version__ = '1.0'

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # # ## pre-training
    # parser.add_argument('--task', dest='TASK', help="task name, one of ['ok']", type=str, default='ok')
    # parser.add_argument('--run_mode', dest='RUN_MODE', help="run mode, one of ['pretrain']", type=str, default='pretrain')
    # parser.add_argument('--cfg', dest='cfg_file', help='config file', type=str, default='configs/pretrain.yml')
    # parser.add_argument('--version', dest='VERSION', help='version name, output folder will be named as version name', type=str, default='pretraining_okvqa')
    # parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default=None)
    # parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    # parser.add_argument('--freeze bert', dest='FRZ_BERT', help='pretrained model path', type=bool, default=False)
    # parser.add_argument('--freeze visual', dest='FRZ_VIS', help='pretrained model path', type=bool, default=False)
    # parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0, 1')
    # parser.add_argument('--gpu0bs', dest='GPU0_BS', help='gpu 0 batch size', type=int, default=38)

    # ## fine-tuing okvqa
    parser.add_argument('--task', dest='TASK', help="task name, one of ['ok', 'coco']", type=str, default='ok')
    parser.add_argument('--run_mode', dest='RUN_MODE', help="run mode, one of ['pretrain', 'finetune', 'finetune_test']", type=str, default='finetune')
    parser.add_argument('--cfg', dest='cfg_file', help='config file', type=str, default='configs/finetune.yml')
    parser.add_argument('--version', dest='VERSION', help='version name, output folder will be named as version name', type=str, default='finetuning_ok')
    parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default='./pretrained/checkpoint.pkl')
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    parser.add_argument('--freeze bert', dest='FRZ_BERT', help='pretrained model path', type=bool, default=False)
    parser.add_argument('--freeze visual', dest='FRZ_VIS', help='pretrained model path', type=bool, default=True)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0')
    parser.add_argument('--gpu0bs', dest='GPU0_BS', help='gpu 0 batch size', type=int, default=40)

    # # ## fine-tuing coco-qa
    # parser.add_argument('--task', dest='TASK', help="task name, one of ['ok', 'coco']", type=str, default='coco')
    # parser.add_argument('--run_mode', dest='RUN_MODE', help="run mode, one of ['pretrain', 'finetune', 'finetune_test']", type=str, default='finetune')
    # parser.add_argument('--cfg', dest='cfg_file', help='config file', type=str, default='configs/finetune.yml')
    # parser.add_argument('--version', dest='VERSION', help='version name, output folder will be named as version name', type=str, default='finetuning_coco')
    # parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default='./pretrained/checkpoint.pkl')
    # parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    # parser.add_argument('--freeze bert', dest='FRZ_BERT', help='pretrained model path', type=bool, default=False)
    # parser.add_argument('--freeze visual', dest='FRZ_VIS', help='pretrained model path', type=bool, default=True)
    # parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='1')
    # parser.add_argument('--gpu0bs', dest='GPU0_BS', help='gpu 0 batch size', type=int, default=40)

    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--large_bert', dest='LARGE_BERT', help='gpu id', type=str, default='./bert-large-uncased')
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=2)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=1)
    args = parser.parse_args()
    return args


def get_runner(__C, evaluater):
    if __C.RUN_MODE == 'pretrain':
        from .pretrain import Runner
    elif __C.RUN_MODE == 'finetune':
        from .finetune import Runner
    elif __C.RUN_MODE == 'finetune_test':
        from .finetune import Runner
    else:
        raise NotImplementedError
    runner = Runner(__C, evaluater)
    return runner