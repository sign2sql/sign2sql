import numpy as np
import argparse
import sys
import os
import torch
import re
import json
import time

from torch.nn.utils import clip_grad_norm

from ..data_utils import data_utils

CKPT_PATTERN = re.compile('^ckpt-(\d+)$')


class Supervisor(object):
	"""
	The base class to manage the high-level model execution processes. The concrete classes for different applications are derived from it.
	"""
	def __init__(self, model, args):
		self.data_processor = data_utils.DataProcessor(args)
		self.model = model
		self.keep_last_n = args.keep_last_n
		self.global_step = 0
		self.batch_size = args.batch_size
		self.model_dir = args.model_dir
		self.args=args


	def load_pretrained(self, load_model):
		print("Read model parameters from %s." % load_model)
		checkpoint = torch.load(load_model)
		self.model.load_state_dict(checkpoint)


	def save_model(self):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		global_step_padded = format(self.global_step, '08d')
		ckpt_name = 'ckpt-' + global_step_padded
		path = os.path.join(self.model_dir, ckpt_name)
		ckpt = self.model.state_dict()
		torch.save(ckpt, path)

		if self.keep_last_n is not None:
			ckpts = []
			for file_name in os.listdir(self.model_dir):
				matched_name = CKPT_PATTERN.match(file_name)
				if matched_name is None or matched_name == ckpt_name:
					continue
				step = int(matched_name.group(1))
				ckpts.append((step, file_name))
			if len(ckpts) > self.keep_last_n:
				ckpts.sort()
				os.unlink(os.path.join(self.model_dir, ckpts[0][1]))


	def train(self, batch_input, batch_output,batch_output_mask):
		#batch_size,seqlen
		self.model.train()
		self.model.optimizer.zero_grad()

		predictions,prediction_result=self.model(batch_input,batch_output,batch_output_mask)

		predictions = predictions[:, 1:, :].reshape(-1, self.args.sql_vocab_size)
		# prediction_result = prediction_result[:, 1:]

		pred_acc = torch.sum(prediction_result == batch_output)
		pred_acc = pred_acc.item() * 1.0 / (batch_output.size()[0] * batch_output.size()[1])
		trg = batch_output[:, 1:].reshape(-1)
		cur_loss=self.model.criteration(predictions,trg)
		cur_loss.backward()
		self.model.optimizer.step()
		return cur_loss,pred_acc

	def eval(self, src_data, data_order_invariant=False, max_eval_size=None):
		self.model.eval()
		data_size = len(src_data)
		if max_eval_size is not None:
			data_size = min(data_size, max_eval_size)
		eval_data = src_data[:data_size]
		test_loss = 0.0
		pred_acc_all = 0.0

		real_prediction = []

		count=0
		for batch_idx in range(0, data_size, self.batch_size):
			batch_input, batch_output,output_mask = self.data_processor.get_batch(eval_data, self.batch_size, batch_idx)
			predictions, prediction_result = self.model(batch_input, batch_output,output_mask,1)

			predictions = predictions[:, 1:, :].reshape(-1, self.args.sql_vocab_size)

			# prediction_result=prediction_result[:,1:]

			pred_acc = torch.sum(prediction_result == batch_output)
			pred_acc_all += pred_acc.item() * 1.0 / (batch_output.size()[0] * batch_output.size()[1])
			trg = batch_output[:, 1:].reshape(-1)
			with torch.no_grad():
				test_loss+= self.model.criteration(predictions, trg)

			cur_predictions = prediction_result.data.cpu().numpy().tolist()

			for i, sample in enumerate(batch_input):
				pred_prog = self.data_processor.ids_to_prog(cur_predictions[i])
				traget_prog=self.data_processor.ids_to_prog(batch_output[i])
				real_prediction.append([pred_prog,traget_prog])
			count+=1

		test_loss /= count

		test_acc = pred_acc_all * 1.0 / count

		return test_loss, test_acc, real_prediction
