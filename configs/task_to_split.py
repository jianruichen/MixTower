# ------------------------------------------------------------------------------ #
__author__ = 'Deguang Chen'
# Description: The goal of this file is to define the mapping from task and data mode to dataset splits.
# ------------------------------------------------------------------------------ #


class DictSafe(dict):

    def __init__(self, data={}):
        dict.__init__(self, data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DictSafe(value)

    def __getitem__(self, key):
        return self.get(key, [])


# TASK_TO_SPLIT[TASK][DATA_MODE]['train_split'] is a list of dataset split name for training
# TASK_TO_SPLIT[TASK][DATA_MODE]['eval_split'] is a list of dataset split name for evaluation
# 'pretrain' mode is used for pretrain, so it does not have 'eval_split'
# 'finetune' mode is used for finetune, heuristics generation and prompting
TASK_TO_SPLIT = {
    'ok': {
        'pretrain': {'train_split': ['v2train', 'v2val', 'vg'],},
        'finetune': {'train_split': ['oktrain'], 'eval_split': ['oktest'],}
    },
    'coco': {
        'pretrain': {'train_split': ['v2train', 'v2val', 'vg'], },
        'finetune': {'train_split': ['cocotrain'], 'eval_split': ['cocotest'], }
    },
}
TASK_TO_SPLIT = DictSafe(TASK_TO_SPLIT)


SPLIT_TO_IMGS = {
    'v2train': 'train2014',
    'v2val': 'val2014',
    'v2valvg_no_ok': 'val2014',
    'vg': 'val2014',
    'oktrain': 'train2014',
    'oktest': 'val2014',
    'cocotrain': 'train2014',
    'cocotest': 'val2014',
}


if __name__ == '__main__':
    print(TASK_TO_SPLIT['okvqa']['test']['train_split'])