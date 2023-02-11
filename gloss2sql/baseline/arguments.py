import argparse
import time
import os
import sys
def get_arg_parser(title):
	parser = argparse.ArgumentParser(description=title)
	parser.add_argument('--cpu', action='store_true', default=False)
	parser.add_argument('--eval', action='store_true',default=True)
	parser.add_argument('--model_dir', type=str, default='./checkpoints/attn_model_layer1_database')
	parser.add_argument('--load_model', type=str, default='./checkpoints/attn_model_layer1_database')
	parser.add_argument('--num_LSTM_layers', type=int, default=1)
	parser.add_argument('--LSTM_hidden_size', type=int, default=512)
	parser.add_argument('--embedding_size', type=int, default=512)
	parser.add_argument('--bidirectional',action='store_true',default=True)

	parser.add_argument('--keep_last_n', type=int, default=None)
	parser.add_argument('--eval_every_n', type=int, default=1500)
	parser.add_argument('--log_interval', type=int, default=1500)
	parser.add_argument('--log_dir', type=str, default='./logs')
	parser.add_argument('--log_name', type=str, default='model_1.csv')

	parser.add_argument('--max_eval_size', type=int, default=1000)

	data_group = parser.add_argument_group('datasets')
	data_group.add_argument('--train_gloss', type=str, default='./random_dataset/data_final_processed/train/train_encode.txt')
	data_group.add_argument('--train_sql', type=str, default='./random_dataset/data_final_processed/train/train_decode.txt')
	data_group.add_argument('--dev_gloss',type=str,default='./random_dataset/data_final_processed/dev/dev_encode.txt')
	data_group.add_argument('--dev_sql',type=str,default='./random_dataset/data_final_processed/dev/dev_decode.txt')
	data_group.add_argument('--test_gloss',type=str,default='./random_dataset/data_final_processed/test/test_encode.txt')
	data_group.add_argument('--test_sql',type=str,default='./random_dataset/data_final_processed/test/test_decode.txt')
	data_group.add_argument('--sql_vocab', type=str, default='./random_dataset/data_final_processed/decode_vocab.txt')
	data_group.add_argument('--gloss_vocab', type=str, default='./random_dataset/data_final_processed/encode_vocab.txt')
	data_group.add_argument('--train_mask', type=str,default='./random_dataset/data_final_processed/train/train_decoder_mask.txt')
	data_group.add_argument('--dev_mask',type=str,default='./random_dataset/data_final_processed/dev/dev_decoder_mask.txt')
	data_group.add_argument('--test_mask', type=str,default='./random_dataset/data_final_processed/test/test_decoder_mask.txt')
	data_group.add_argument('--gloss_vocab_size', type=int, default=None)
	data_group.add_argument('--sql_vocab_size', type=int, default=None)
	data_group.add_argument('--joint_plot_types', action='store_true', default=False)
	data_group.add_argument('--data_order_invariant', action='store_true', default=False)
	data_group.add_argument('--max_gloss_len', type=int, default=512)
	data_group.add_argument('--max_decode_len', type=int, default=200)

	model_group = parser.add_argument_group('model')
	model_group.add_argument('--attention_mechanism',action='store_true',default=True)

	train_group = parser.add_argument_group('train')
	train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
	train_group.add_argument('--lr', type=float, default=1e-3)
	train_group.add_argument('--dropout_rate', type=float, default=0)
	train_group.add_argument('--num_epochs', type=int, default=50)
	train_group.add_argument('--batch_size', type=int, default=128)
	train_group.add_argument('--param_init', type=float, default=0.1)
	train_group.add_argument('--seed', type=int, default=None)

	return parser