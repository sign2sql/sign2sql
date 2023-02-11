import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sign2sql.utils.utils_sign2sql import *
from sign2sql.models.slt import *
from utils.metrics import bleu, chrf, rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_infer', default=False, action='store_true')
    parser.add_argument('--infer_loop', default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')

    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    # parser.add_argument('--fine_tune',
    #                     default=False,
    #                     action='store_true',
    #                     help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='SLT1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=3, type=int, help="The number of Transformer layers in Encoder or Decoder.")
    parser.add_argument('--dr', default=0.1, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=256, type=int, help="The dimension of hidden vector in the SLTModel.")

    # # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    # parser.add_argument('--EG',
    #                     default=False,
    #                     action='store_true',
    #                     help="If present, Execution guided decoding is used in test.")
    # parser.add_argument('--beam_size',
    #                     type=int,
    #                     default=4,
    #                     help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def get_opt(model):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=0)
    return opt


def get_models(args, BERT_PT_PATH, trained=False, path_model=None):
    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: False")
    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)

    # Get SLT model
    model = SLTModel(model_bert.embeddings, args.hS, args.lr, args.lS)
    model = model.to(device)

    if trained:
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])
    
    return model, tokenizer


def get_data(path_sign2text, args):
    train_data, dev_data, table = load_sign2text(path_sign2text)
    train_loader, dev_loader = get_loader_sign2text(train_data, dev_data, args.bS, shuffle_train=True)
    
    return train_data, dev_data, table, train_loader, dev_loader


def train(train_loader, model, opt, tokenizer, 
          accumulate_gradients=1, st_pos=0):
    model.train()

    ave_loss = 0
    cnt = 0  # count the # of examples

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        nlu, videos = get_fields_sign2text(t)
        # nlu : natural language
        # videos : video npys

        video_array, video_array_mask = get_padded_batch_video(videos)
        input_text_array, output_text_array, text_mask_array = get_input_output_token(tokenizer, nlu)
        # video_array: [B, T_video, 144, 144]
        # input_text_array: [B, T_text]

        # Forward
        y_pred = model(video_array, video_array_mask, input_text_array, text_mask_array)

        # Calculate loss & step
        loss = compute_ce_loss(y_pred, output_text_array, text_mask_array)

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Statistics
        ave_loss += loss.item()
    
    ave_loss /= cnt
    acc = [ave_loss]
    aux_out = 1
    
    return acc, aux_out


def test(data_loader, model, tokenizer, 
         st_pos=0, beam_size=5):
    model.eval()
    
    cnt = 0
    all_text_pred = []
    all_text_trg = []
    for iB, t in enumerate(data_loader):
    
        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, videos = get_fields_sign2text(t)
        # nlu : natural language
        # videos : video npys

        video_array, video_array_mask = get_padded_batch_video(videos)
        input_text_array, output_text_array, text_mask_array = get_input_output_token(tokenizer, nlu)
        # video_array: [B, T_video, 144, 144]
        # input_text_array: [B, T_text]

        # Inference SLT Beam
        stacked_txt_output = inference_slt_beam(model, tokenizer, video_array, video_array_mask, beam_size)
        texts = []
        pad_id = 0
        for ts in stacked_txt_output:
            if pad_id in ts:
                ids = ts[:ts.tolist().index(pad_id)][:-1]  # remove <PAD> & <EOS>
            else:
                ids = ts[:-1]  # remove <EOS>
            tokens = tokenizer.convert_ids_to_tokens(ids)
            cur = tokenizer.concat_tokens(tokens)
            texts.append(cur)
        all_text_pred += texts
        
        text_trg = []
        output_text_array, text_mask_array = output_text_array.cpu(), text_mask_array.cpu()
        for i in range(output_text_array.size(0)):
            ids = output_text_array[i, text_mask_array[i]][:-1]  # remove <PAD> & <EOS>
            tokens = tokenizer.convert_ids_to_tokens(ids)
            cur = tokenizer.concat_tokens(tokens)
            text_trg.append(cur)
        all_text_trg += text_trg
    
    # Calculate BLEU scores
    bleus = bleu(references=all_text_trg, hypotheses=all_text_pred)
    results = {}
    results['bleu1'] = bleus['bleu1']
    results['bleu2'] = bleus['bleu2']
    results['bleu3'] = bleus['bleu3']
    results['bleu4'] = bleus['bleu4']
    results['chrf'] = chrf(references=all_text_trg, hypotheses=all_text_pred)
    results['rouge'] = rouge(references=all_text_trg, hypotheses=all_text_pred)
    return results, all_text_pred


def print_result(epoch, print_data, dname):
    print(f'{dname} results ------------')
    print(f" Epoch: {epoch}")
    print(print_data)

def save_text_pred_dev(save_root, all_text_pred, dname):
    with open(os.path.join(save_root, dname), 'w') as f:
        for text in all_text_pred:
            f.write(text+'\n')


if __name__ == '__main__':
    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    BERT_PT_PATH = '/mnt/gold/zsj/data/sign2sql/model/annotated_wikisql_and_PyTorch_bert_param'
    path_sign2text = '/mnt/gold/zsj/data/sign2sql/dataset'

    path_save_for_evaluation = '/mnt/gold/zsj/data/sign2sql/model/sign2text_save'
    if not os.path.exists(path_save_for_evaluation):
        os.mkdir(path_save_for_evaluation)
    
    ## 3. Load data

    train_data, dev_data, table, train_loader, dev_loader = get_data(path_sign2text, args)

    ## 4. Build & Load models
    if not args.trained:
        model, tokenizer = get_models(args, BERT_PT_PATH)
    else:
        # To start from the pre-trained models, un-comment following lines.
        path_model = os.path.join(path_save_for_evaluation, 'model_best.pt')
        model, tokenizer = get_models(args, BERT_PT_PATH, 
                                      trained=True, path_model=path_model)

    ## 5. Get optimizers
    if args.do_train:
        opt = get_opt(model)

        ## 6. Train
        ROUGE_best = -1
        epoch_best = -1
        for epoch in range(args.tepoch):
            # train
            acc_train, aux_out_train = train(train_loader,
                                             model,
                                             opt,
                                             tokenizer,
                                             args.accumulate_gradients,
                                             st_pos=0)
            
            # check DEV
            with torch.no_grad():
                results, all_text_pred_dev = test(dev_loader,
                                                  model,
                                                  tokenizer,
                                                  st_pos=0,
                                                  beam_size=5)

            print_result(epoch, acc_train, 'train')
            print_result(epoch, results, 'dev')
            
            # save pred text on dev
            save_text_pred_dev(path_save_for_evaluation, all_text_pred_dev, 'all_text_pred_dev.txt')

            # save best model
            # Based on Dev Set ROUGE score
            rouge_score = results['rouge']
            if rouge_score > ROUGE_best:
                ROUGE_best = rouge_score
                epoch_best = epoch
                # save best model
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join(path_save_for_evaluation, 'model_best.pt'))
            
            print(f" Best Dev ROUGE score: {ROUGE_best} at epoch: {epoch_best}")

