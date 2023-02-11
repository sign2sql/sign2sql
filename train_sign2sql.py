# pretrained video embedding from SLT
# fix case uncase bug in generated sql
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

from utils.metrics import bleu, chrf, rouge

from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

from sign2sql.utils.utils_sign2sql import *
from sign2sql.models.sign2sql_v2 import *
from sign2sql.models.slt import SLTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--model_save_path", default='../sign2sql_v2_save', type=str)
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_infer', default=False, action='store_true')
    parser.add_argument('--infer_loop', default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')

    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
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

    # 1.3.1 Sign2SQL Encoder module parameters
    parser.add_argument('--lS_enc', default=2, type=int, help="The number of Transformer layers in Encoder or Decoder.")
    parser.add_argument('--dr_enc', default=0.0, type=float, help="Dropout rate.")
    parser.add_argument('--lr_enc', default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--hS_enc", default=256, type=int, help="The dimension of hidden vector in the SLTModel.")

    # 1.3.2 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

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


def get_opt(model, model_encoder):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=0)
    opt_encoder = torch.optim.Adam(filter(lambda p: p.requires_grad, model_encoder.parameters()),
                                lr=args.lr_enc, weight_decay=0)

    return opt, opt_encoder


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model_slt=None, path_model_encoder=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion
    
    # Get SLT model
    model_slt = SLTModel(model_bert.embeddings, 256, 1e-4, 3)  # TODO: train sign2text with dropout 0.1
    model_slt = model_slt.to(device)
    # load pretrained slt pt !!!
    assert path_model_slt != None
    if torch.cuda.is_available():
        res = torch.load(path_model_slt)
    else:
        res = torch.load(path_model_slt, map_location='cpu')
    model_slt.load_state_dict(res['model'])

    # Get Encoder model
    model_encoder = Sign2SQLv2Model(model_slt, model_bert.embeddings, args.hS_enc, args.iS, args.dr_enc, args.lS_enc)
    model_encoder = model_encoder.to(device)

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        # load sql decoder
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

        # load sign2sql encoder
        assert path_model_encoder != None
        if torch.cuda.is_available():
            res = torch.load(path_model_encoder)
        else:
            res = torch.load(path_model_encoder, map_location='cpu')

        model_encoder.load_state_dict(res['model'])

    return model_encoder, model, model_bert, tokenizer, bert_config



# modify by zsj for debugging
def get_data(path_wikisql, args):
    train_data, dev_data, table = load_sign2text(path_wikisql)

    train_data = train_data[:12]

    train_loader, dev_loader = get_loader_sign2text(train_data, dev_data, args.bS, shuffle_train=True)
    train_table = dev_table = table
    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_encoder, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_encoder=None, path_db=None, db_name='all'):
    model.train()
    model_encoder.train()

    ave_loss = 0
    ave_loss1 = 0
    ave_loss2 = 0
    cnt = 0  # count the # of examples

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.
        _, videos = get_fields_sign2text(t)
        # nlu : natural language
        # videos : video npys
        video_array, video_array_mask = get_padded_batch_video(videos)

        input_where_array, output_where_array, where_mask_array = get_input_output_where_ids(tokenizer, sql_i)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        y_pred, wemb_v, wemb_h, l_n, l_hpu, l_hs, header_input_ids, header_input_mask = \
            get_wemb_sign2sql_encoder(model_encoder, tokenizer, video_array, video_array_mask, 
                hds, input_where_array, where_mask_array)
        # y_pred: predicted where tokens
        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        
        # Calculate loss & step
        loss_where = compute_ce_loss(y_pred, output_where_array, where_mask_array)

        # score
        s_sc, s_sa, s_wn = model.forward_no_where(wemb_v, l_n, wemb_h, l_hpu, l_hs,
                                                  g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=None, g_wvi=None)

        # Calculate loss & step
        loss_structed = Loss_sw_se_no_where(s_sc, s_sa, s_wn, g_sc, g_sa, g_wn)

        loss = loss_where + loss_structed

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_encoder:
                opt_encoder.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_encoder:
                    opt_encoder.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_encoder:
                opt_encoder.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # statistics
        ave_loss += loss.item()
        ave_loss1 += loss_where.item()
        ave_loss2 += loss_structed.item()

    ave_loss /= cnt
    ave_loss1 /= cnt
    ave_loss2 /= cnt
    acc = [ave_loss, ave_loss1, ave_loss2]

    aux_out = 1

    return acc, aux_out


# test sign2sql utils
def generate_sql_from_hybrid(pr_sc, pr_sa, stacked_txt_output, tokenizer, headers):
    pr_sql_i = []
    pad_id = 0
    num_of_cond_ops = 4  # cond_ops = ['=', '>', '<', 'OP']
    for b, ts in enumerate(stacked_txt_output):
        if pad_id in ts:
            ids = ts[:ts.tolist().index(pad_id)]  # remove <PAD>
        else:
            ids = ts  # 
        cond_ids_list = []
        tmp = []
        for id1 in ids:
            cur_tok = tokenizer.ids_to_tokens[id1]
            if cur_tok == '[CLS]':
                cond_ids_list.append(tmp)
                tmp = []
            elif cur_tok == '[SEP]':
                cond_ids_list.append(tmp)
                break
            else:
                tmp.append(id1)
        
        num_of_headers_i = len(headers[b])
        cond_list = []
        for cond_ids in cond_ids_list:
            tokens = tokenizer.convert_ids_to_tokens(cond_ids)
            cur = tokenizer.concat_tokens(tokens).split()
            # cur: ["2", "0", "ABC", "AB"]
            if len(cur) <= 2: continue
            if not cur[0].isdigit() or not cur[1].isdigit(): continue
            wc = max(min(int(cur[0]), num_of_headers_i-1), 0)
            wo = max(min(int(cur[1]), num_of_cond_ops-1), 0)
            cond_list.append([wc, wo, ' '.join(cur[2:])])
        
        pr_sql_i1 = {'agg': pr_sa[b], 'sel': pr_sc[b], 'conds': cond_list}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i


# fixing where value case bug
def change_sql_case(sql_i, pr_sql_i):
    fixed_pr_sql_i = []
    for b in range(len(sql_i)):
        gt = sql_i[b]
        pr = pr_sql_i[b]
        gt_conds = gt['conds']
        pr_conds = pr['conds']
        fixed_pr_conds = []
        for cond in pr_conds:
            flag = 0
            for gc in gt_conds:
                if str(gc[-1]).lower() == cond[-1].lower():
                    # fixed_pr_conds.append([cond[0], cond[1], gc[-1]])
                    fixed_pr_conds.append(gc)  # test gc !
                    flag = 1
                    break
            if flag == 0:
                fixed_pr_conds.append(cond)
        cur = {'agg': pr['agg'], 'sel': pr['sel'], 'conds': fixed_pr_conds}
        fixed_pr_sql_i.append(cur)
    return fixed_pr_sql_i


def test(data_loader, data_table, model, model_encoder, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, db_name='all'):
    model.eval()
    model_encoder.eval()

    all_sql_query_pred = []
    all_sql_query_trg = []
    ave_loss = 0
    cnt = 0
    # cnt_sc = 0
    # cnt_sa = 0
    # cnt_wn = 0
    # cnt_wc = 0
    # cnt_wo = 0
    # cnt_wv = 0
    # cnt_wvi = 0
    # cnt_lx = 0  # to be added
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        _, videos = get_fields_sign2text(t)
        # nlu : natural language
        # videos : video npys
        video_array, video_array_mask = get_padded_batch_video(videos)

        input_where_array, output_where_array, where_mask_array = get_input_output_where_ids(tokenizer, sql_i)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)        
        
        y_pred, wemb_v, wemb_h, l_n, l_hpu, l_hs, header_input_ids, header_input_mask = \
            get_wemb_sign2sql_encoder(model_encoder, tokenizer, video_array, video_array_mask, 
                hds, input_where_array, where_mask_array)
        # try:
        #     g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        #     g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # except:
        #     # Exception happens when where-condition is not found in nlu_tt.
        #     # In this case, that train example is not used.
        #     # During test, that example considered as wrongly answered.
        #     for b in range(len(nlu)):
        #         results1 = {}
        #         results1["error"] = "Skip happened"
        #         results1["nlu"] = nlu[b]
        #         results1["table_id"] = tb[b]["id"]
        #         results.append(results1)
        #     continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn = model.forward_no_where(wemb_v, l_n, wemb_h, l_hpu, l_hs,
                                                  g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=None, g_wvi=None)

            # get loss & step
            loss_structed = Loss_sw_se_no_where(s_sc, s_sa, s_wn, g_sc, g_sa, g_wn)

            # prediction
            pr_sc, pr_sa, pr_wn = pred_sw_se_no_where(s_sc, s_sa, s_wn)
            
            stacked_txt_output = inference_where_beam(model_encoder, tokenizer, video_array, video_array_mask, header_input_ids, header_input_mask, beam_size)

            pr_sql_i = generate_sql_from_hybrid(pr_sc, pr_sa, stacked_txt_output, tokenizer, hds)
            pr_sql_i = change_sql_case(sql_i, pr_sql_i)  # fix case bug !!!
        else:
            # not need
            pass

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        all_sql_query_trg += g_sql_q
        all_sql_query_pred += pr_sql_q

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        # cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        # cnt_wc1_list, cnt_wo1_list, \
        # cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
        #                                               pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
        #                                               sql_i, pr_sql_i,
        #                                               mode='test')

        # cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
        #                                cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
     
        # stat
        # ave_loss += loss.item()

        # count
        # cnt_sc += sum(cnt_sc1_list)
        # cnt_sa += sum(cnt_sa1_list)
        # cnt_wn += sum(cnt_wn1_list)
        # cnt_wc += sum(cnt_wc1_list)
        # cnt_wo += sum(cnt_wo1_list)
        # cnt_wv += sum(cnt_wv1_list)
        # cnt_wvi += sum(cnt_wvi1_list)
        # cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        # current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        # cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
        #              cnt_x1_list]
        # cnt_list.append(cnt_list1)
        # # report
        # if detail:
        #     report_detail(hds, nlu,
        #                   g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
        #                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
        #                   cnt_list1, current_cnt)

    # Calculate sign2text2sql scores
    bleus2 = bleu(references=all_sql_query_trg, hypotheses=all_sql_query_pred)
    results_acc = {}
    results_acc['bleu1'] = bleus2['bleu1']
    results_acc['bleu2'] = bleus2['bleu2']
    results_acc['bleu3'] = bleus2['bleu3']
    results_acc['bleu4'] = bleus2['bleu4']
    results_acc['chrf'] = chrf(references=all_sql_query_trg, hypotheses=all_sql_query_pred)
    results_acc['rouge'] = rouge(references=all_sql_query_trg, hypotheses=all_sql_query_pred)

    
    # ave_loss /= cnt
    # acc_sc = cnt_sc / cnt
    # acc_sa = cnt_sa / cnt
    # acc_wn = cnt_wn / cnt
    # acc_wc = cnt_wc / cnt
    # acc_wo = cnt_wo / cnt
    # acc_wvi = cnt_wvi / cnt
    # acc_wv = cnt_wv / cnt
    # acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt
    results_acc['execution accuracy'] = acc_x

    # acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    
    return results_acc, results, all_sql_query_pred, cnt_list

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
    path_sign2sql = '/mnt/gold/zsj/data/sign2sql/dataset'
    SLT_PT_PATH = '/mnt/gold/zsj/data/sign2sql/model/sign2text_save/model_best.pt'

    path_save_for_evaluation = args.model_save_path  # default='../sign2sql_v2_save'
    if not os.path.exists(path_save_for_evaluation):
        os.mkdir(path_save_for_evaluation)

    ## 3. Load data

    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_sign2sql, args)
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    if not args.trained:
        model_encoder, model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, path_model_slt=SLT_PT_PATH)
    else:
        # To start from the pre-trained models, un-comment following lines.
        # no finetune bert
        path_model_bert = os.path.join(path_save_for_evaluation, 'model_bert_best.pt')
        path_model = os.path.join(path_save_for_evaluation, 'model_best.pt')
        path_model_encoder = os.path.join(path_save_for_evaluation, 'model_encoder_best.pt')
        model_encoder, model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                        path_model_bert=path_model_bert, path_model_slt=SLT_PT_PATH, path_model_encoder=path_model_encoder, path_model=path_model)

    ## 5. Get optimizers
    if args.do_train:
        opt, opt_encoder = get_opt(model, model_encoder)

        ## 6. Train
        acc_x_t_best = -1
        epoch_best = -1
        for epoch in range(args.tepoch):
            # train
            acc_train, aux_out_train = train(train_loader,
                                             train_table,
                                             model,
                                             model_encoder,
                                             opt,
                                             bert_config,
                                             tokenizer,
                                             args.max_seq_length,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_encoder=opt_encoder,
                                             st_pos=0,
                                             path_db=path_sign2sql,
                                             db_name='all')

            # check DEV
            with torch.no_grad():
                results_acc, results, all_sql_query_pred, cnt_list = test(dev_loader,
                                                                        dev_table,
                                                                        model,
                                                                        model_encoder,
                                                                        bert_config,
                                                                        tokenizer,
                                                                        args.max_seq_length,
                                                                        args.num_target_layers,
                                                                        detail=False,
                                                                        path_db=path_sign2sql,
                                                                        st_pos=0,
                                                                        db_name='all', EG=args.EG)

            print_result(epoch, acc_train, 'train')
            print_result(epoch, results_acc, 'dev')

            # save results for the official evaluation
            save_for_evaluation(path_save_for_evaluation, results, 'dev')
            save_text_pred_dev(path_save_for_evaluation, all_sql_query_pred, 'text2sql_all_sql_query_pred.txt')

            # save best model
            acc_x_t = results_acc['execution accuracy']
            if acc_x_t > acc_x_t_best:
                acc_x_t_best = acc_x_t
                epoch_best = epoch
                # save best model
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join(path_save_for_evaluation, 'model_best.pt'))

                state = {'model': model_encoder.state_dict()}
                torch.save(state, os.path.join(path_save_for_evaluation, 'model_encoder_best.pt'))

                state = {'model_bert': model_bert.state_dict()}
                torch.save(state, os.path.join(path_save_for_evaluation, 'model_bert_best.pt'))

            print(f" Best Dev x acc: {acc_x_t_best} at epoch: {epoch_best}")
