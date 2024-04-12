import argparse
import yaml
import torch
torch.cuda.is_available()

from evaluation.okvqa_evaluate import OKEvaluater
from configs.task_cfgs import Cfgs
from mixtower import get_args, get_runner

# parse cfgs and args
args = get_args()
__C = Cfgs(args)
with open(args.cfg_file, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
__C.override_from_dict(yaml_dict)
print(__C)

# build runner
if __C.RUN_MODE == 'pretrain':
    evaluater = None
else:
    evaluater = OKEvaluater(__C.EVAL_ANSWER_PATH, __C.EVAL_QUESTION_PATH,)

runner = get_runner(__C, evaluater)
runner.run()


