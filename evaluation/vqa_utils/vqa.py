__author__ = 'aagrawal'
__version__ = '0.9'

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy


class VQA:
	def __init__(self, annotation_file=None, question_file=None):
		# load dataset
		self.dataset = {}
		self.questions = {}
		self.qa = {}
		self.qqa = {}
		self.imgToQA = {}
		if not annotation_file == None and not question_file == None:
			print('loading VQA annotations and questions into memory...')
			time_t = datetime.datetime.utcnow()
			dataset = json.load(open(annotation_file, 'r', encoding='utf-8'))
			questions = json.load(open(question_file, 'r', encoding='utf-8'))
			print(datetime.datetime.utcnow() - time_t)
			self.dataset = dataset
			self.questions = questions
			self.createIndex()

	def createIndex(self):
		# create index
		print('creating index...')
		imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
		qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
		qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
		for ann in self.dataset['annotations']:
			imgToQA[ann['image_id']] += [ann]
			qa[ann['question_id']] = ann
		for ques in self.questions['questions']:
			qqa[ques['question_id']] = ques
		print('index created!')

		# create class members
		self.qa = qa
		self.qqa = qqa
		self.imgToQA = imgToQA

	def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
		quesIds = quesIds if type(quesIds) == list else [quesIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

		if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(quesIds) == 0:
				anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa], [])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['image_id'] for ann in anns]
		return ids

	def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
		imgIds = imgIds if type(imgIds) == list else [imgIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

		if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(imgIds) == 0:
				anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA], [])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['question_id'] for ann in anns]
		return ids

	def loadRes(self, resFile, quesFile):
		res = VQA()
		res.questions = json.load(open(quesFile))
		# res.dataset['info'] = copy.deepcopy(self.questions['info'])
		# res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
		# res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
		# res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
		# res.dataset['license'] = copy.deepcopy(self.questions['license'])

		print('Loading and preparing results...     ')
		time_t = datetime.datetime.utcnow()
		anns = json.load(open(resFile))
		assert type(anns) == list, 'results is not an array of objects'
		annsQuesIds = [ann['question_id'] for ann in anns]
		assert set(annsQuesIds) == set(self.getQuesIds()), 'Results do not correspond to current VQA set.'
		for ann in anns:
			quesId = ann['question_id']
			# if res.dataset['task_type'] == 'Multiple Choice':
			# 	assert ann['answer'] in self.qqa[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
			qaAnn = self.qa[quesId]
			ann['image_id'] = qaAnn['image_id']
			ann['question_type'] = qaAnn['question_type']
			ann['answer_type'] = qaAnn['answer_type']
		print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))

		res.dataset['annotations'] = anns
		res.createIndex()
		return res

	def loadQA(self, ids=[]):
		if type(ids) == list:
			return [self.qa[id] for id in ids]
		elif type(ids) == int:
			return [self.qa[ids]]

	def showQA(self, anns):
		if len(anns) == 0:
			return 0
		for ann in anns:
			quesId = ann['question_id']
			print("Question: %s" % (self.qqa[quesId]['question']))
			for ans in ann['answers']:
				print("Answer %d: %s" % (ans['answer_id'], ans['answer']))

	def info(self):
		# Print information about the VQA annotation file.
		for key, value in self.dataset['info'].items():
			print('%s: %s' % (key, value))

