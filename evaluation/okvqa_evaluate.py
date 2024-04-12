# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Evaluation script for OK-VQA
# ------------------------------------------------------------------------------ #

import json
from evaluation.vqa_utils.vqa import VQA
from evaluation.vqa_utils.vqaEval import VQAEval
from .ans_punct import prep_ans
import argparse


def _evaluate(annotation_file: str, question_file: str, result_file: str):
    vqa = VQA(annotation_file, question_file)
    vqaRes_prophet = vqa.loadRes(result_file, question_file)
    vqaEval_prophet = VQAEval(vqa, vqaRes_prophet, n=2)
    vqaEval_prophet.evaluate()

    question_types = {
        "0": "Object",
        "1": "Number",
        "2": "Color",
        "3": "Location",
    }

    result_str = ''
    result_str += "Overall Accuracy is: %.02f\n" % (vqaEval_prophet.accuracy['overall'])
    # result_str += f"{'Question Type':40s}\t{'Prophet'}\n"
    # for quesType in question_types:
    #     result_str += "%-40s\t%.02f\n" % (question_types[quesType], vqaEval_prophet.accuracy['perQuestionType'][quesType])
    return result_str


class OKEvaluater:
    def __init__(self, annotation_path: str, question_path: str):
        self.annotation_path = annotation_path
        self.question_path = question_path
        self.result_file = []
        self.result_path = None

    def init(self):
        self.result_file = []
    
    def add(self, qid, answer):
        qid = int(qid)
        self.result_file.append({'question_id': qid, 'answer': answer})
    
    def save(self, result_path: str):
        self.result_path = result_path
        json.dump(self.result_file, open(self.result_path, 'w'))
    
    def evaluate(self, logfile=None):
        assert self.result_path is not None, "Please save the result file first."
        eval_str = _evaluate(self.annotation_path, self.question_path, self.result_path)
        print()
        print(eval_str)
        if logfile is not None:
            print(eval_str + '\n', file=logfile)

    def prep_ans(self, answer):
        return prep_ans(answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OK-VQA result file.')
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--question_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()
    result_str = _evaluate(args.annotation_path, args.question_path, args.result_path)
    print(result_str)