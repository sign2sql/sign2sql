import random
import torch
import numpy as np

import Sign2Code.models.data_utils.data_utils as data_utils
import Sign2Code.arguments as arguments
# from Sign2Code.models.model import Encoder,Decoder,Seq2Seq
import Sign2Code.models.model_utils.supervisor as model_utils
import os
from Sign2Code.models.new_model import Encoder,ATT_Decoder,Decoder,Seq2Seq,Attention
from nltk.translate.bleu_score import corpus_bleu
from Sign2Code.tools import EarlyStopping
def create_model(args):
	encoder=Encoder(args)
	if args.attention_mechanism:
		attention=Attention(args)
		decoder = ATT_Decoder(args,attention)
	else:
		decoder=Decoder(args)
	model = Seq2Seq(args,encoder,decoder)
	if model.cuda_flag:
		model=model.cuda()
	model_supervisor=model_utils.Supervisor(model,args)
	if args.load_model:
		model_supervisor.load_pretrained(args.load_model)
	else:
		print('Created model with fresh parameters.')
		model_supervisor.model.init_weights(args.param_init)
	return model_supervisor

def train(args):
	print('Training:')

	data_processor=data_utils.DataProcessor(args)
	train_gloss_data,train_sql_data,train_mask_data=data_processor.load_data(args.train_gloss,args.train_sql,args.train_mask)
	train_data,train_indices=data_processor.preprocess(train_gloss_data,train_sql_data,train_mask_data)
	dev_gloss_data,dev_sql_data,dev_mask_data=data_processor.load_data(args.dev_gloss,args.dev_sql,args.dev_mask)
	dev_data,dev_indices=data_processor.preprocess(dev_gloss_data,dev_sql_data,dev_mask_data)

	train_data_size=len(train_data)
	args.gloss_vocab_size=data_processor.gloss_vocab_size
	args.sql_vocab_size=data_processor.sql_vocab_size

	model_supervisor=create_model(args)

	max_acc = 0
	early_stopping = EarlyStopping(patience=7, verbose=True,path=args.model_dir)
	for epoch in range(args.num_epochs):
		random.shuffle(train_data)
		epoch_loss=0.0
		epoch_acc=0.0
		count=0
		for batch_idx in range(0,train_data_size,args.batch_size):
			# print(epoch,batch_idx)
			batch_input, batch_output,batch_output_mask = data_processor.get_batch(train_data, args.batch_size, batch_idx)
			train_loss ,train_acc= model_supervisor.train(batch_input, batch_output,batch_output_mask)
			epoch_loss+=train_loss
			epoch_acc+=train_acc
			count+=1

		train_loss=epoch_loss / count
		print('train loss: %.4f train acc: %.4f' % (epoch_loss / count, epoch_acc / count))


		eval_loss, eval_acc, pred= model_supervisor.eval(dev_data, args.data_order_invariant, args.max_eval_size)
		early_stopping(eval_acc, model_supervisor.model)
		if early_stopping.early_stop:
			print("Early stopping")
			break

		print('dev loss: %.4f dev acc: %.4f' % (eval_loss, eval_acc))


def evaluate(args):
	print('Evaluation')
	data_processor = data_utils.DataProcessor(args)
	test_gloss_data, test_sql_data, test_mask_data = data_processor.load_data(args.test_gloss, args.test_sql,args.test_mask)
	test_data, test_indices = data_processor.preprocess(test_gloss_data, test_sql_data,test_mask_data)

	args.gloss_vocab_size = data_processor.gloss_vocab_size
	args.sql_vocab_size = data_processor.sql_vocab_size
	model_supervisor = create_model(args)

	test_loss,test_acc, predictions = model_supervisor.eval(test_data, args.data_order_invariant)
	print('test loss: %.4f test acc: %.4f' % (test_loss, test_acc))
	# print(predictions[:100])
	acc=0.0
	for prediction in predictions:
		pred=prediction[0][1:-1]
		target=prediction[1][1:-1]
		print(' '.join(pred))
		pred_acc=0
		for i in target:
			if i in pred:
				pred_acc+=1
		acc+=(pred_acc * 1.0 )/ (len(target))
	print(acc/len(predictions))
	for i in range(4):
		n=i+1
		weights = [1 / n] * n + [0] * (4 - n)
		preds = [p[0][1:-1] for p in predictions]
		targets = [[t[1][1:-1]] for t in predictions]
		print(corpus_bleu(targets, preds, weights=weights))




if __name__ == "__main__":
	arg_parser = arguments.get_arg_parser('gloss2sql')
	args = arg_parser.parse_args()
	args.cuda = not args.cpu and torch.cuda.is_available()
	if args.cuda:
		torch.cuda.set_device(3)
	random.seed(args.seed)
	np.random.seed(args.seed)
	if args.eval:
		evaluate(args)
	else:
		train(args)
