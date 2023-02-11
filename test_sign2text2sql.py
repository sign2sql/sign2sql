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

from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--path_save_for_evaluation", default="../sign2text2sql_test_v2", type=str)
    # parser.add_argument("--do_train", default=False, action='store_true')
    # parser.add_argument('--do_infer', default=False, action='store_true')
    # parser.add_argument('--infer_loop', default=False, action='store_true')
    
    parser.add_argument("--separate_test", default=False, action='store_true')  # test sign2text and text2sql separately ? default false
    parser.add_argument("--trained", default=True, action='store_true')

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

    # 1.3.1 SLT module parameters
    parser.add_argument('--lS_slt', default=3, type=int, help="The number of Transformer layers in Encoder or Decoder.")
    parser.add_argument('--dr_slt', default=0.1, type=float, help="Dropout rate.")
    parser.add_argument('--lr_slt', default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--hS_slt", default=256, type=int, help="The dimension of hidden vector in the SLTModel.")

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


# get sign2text & text2sql models
def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model_sign2text=None, path_model_text2sql=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: False")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get SLT model
    model_slt = SLTModel(model_bert.embeddings, args.hS_slt, args.lr_slt, args.lS_slt)
    model_slt = model_slt.to(device)

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model_text2sql = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model_text2sql = model_text2sql.to(device)

    if trained:
        # load text2sql
        assert path_model_bert != None
        assert path_model_text2sql != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model_text2sql)
        else:
            res = torch.load(path_model_text2sql, map_location='cpu')

        model_text2sql.load_state_dict(res['model'])
        
        # load sign2text
        assert path_model_sign2text != None

        if torch.cuda.is_available():
            res = torch.load(path_model_sign2text)
        else:
            res = torch.load(path_model_sign2text, map_location='cpu')

        model_slt.load_state_dict(res['model'])

    return model_slt, model_text2sql, model_bert, tokenizer, bert_config


def get_data(path_sign2sql, args):
    test_data, table = load_sign2sql_test(path_sign2sql)
    test_loader = get_loader_sign2sql_test(test_data, args.bS, shuffle_test=False)
    
    return test_data, table, test_loader


def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')


def test(data_loader, data_table, model_slt, model_text2sql, model_bert, bert_config, tokenizer, 
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, db_name='all', separate_test=False):
    # separate_test: test SLT and text2sql separately
    model_slt.eval()
    model_text2sql.eval()
    model_bert.eval()
    
    # sign2text
    all_text_pred = []
    all_text_trg = []
    # text2sql
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
    # cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))
    text2sql_mid_results = []
    for iB, t in enumerate(data_loader):
        
        cnt += len(t)
        if cnt < st_pos:
            continue
        # 1. SLT
        # Get fields
        nlu, videos = get_fields_sign2text(t)
        # nlu : natural language
        # videos : video npys

        video_array, video_array_mask = get_padded_batch_video(videos)
        input_text_array, output_text_array, text_mask_array = get_input_output_token(tokenizer, nlu)
        # video_array: [B, T_video, 144, 144]
        # input_text_array: [B, T_text]

        # Inference SLT Beam
        stacked_txt_output = inference_slt_beam(model_slt, tokenizer, video_array, video_array_mask, beam_size)
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

        # generate nlu_t_pred
        nlu_pred = texts
        nlu_t_pred = [t.split() for t in texts]

        # 2. text to sql
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        # replace text
        if not separate_test:
            nlu, nlu_t = nlu_pred, nlu_t_pred

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                text2sql_mid_results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model_text2sql(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # get loss & step
            loss = torch.tensor([0])
            # loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model_text2sql.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                            l_hs, engine, tb,
                                                                            nlu_t, nlu_tt,
                                                                            tt_to_t_idx, nlu,
                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None
            loss = torch.tensor([0])

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
            text2sql_mid_results.append(results1)

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
        ave_loss += loss.item()

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
        # report
        # if detail:
        #     report_detail(hds, nlu,
        #                   g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
        #                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
        #                   cnt_list1, current_cnt)

    # Calculate SLT BLEU scores
    bleus = bleu(references=all_text_trg, hypotheses=all_text_pred)
    results_slt = {}
    results_slt['bleu1'] = bleus['bleu1']
    results_slt['bleu2'] = bleus['bleu2']
    results_slt['bleu3'] = bleus['bleu3']
    results_slt['bleu4'] = bleus['bleu4']
    results_slt['chrf'] = chrf(references=all_text_trg, hypotheses=all_text_pred)
    results_slt['rouge'] = rouge(references=all_text_trg, hypotheses=all_text_pred)
    
    # Calculate sign2text2sql scores
    bleus2 = bleu(references=all_sql_query_trg, hypotheses=all_sql_query_pred)
    results_text2sql = {}
    results_text2sql['bleu1'] = bleus2['bleu1']
    results_text2sql['bleu2'] = bleus2['bleu2']
    results_text2sql['bleu3'] = bleus2['bleu3']
    results_text2sql['bleu4'] = bleus2['bleu4']
    results_text2sql['chrf'] = chrf(references=all_sql_query_trg, hypotheses=all_sql_query_pred)
    results_text2sql['rouge'] = rouge(references=all_sql_query_trg, hypotheses=all_sql_query_pred)

    ave_loss /= cnt
    # acc_sc = cnt_sc / cnt
    # acc_sa = cnt_sa / cnt
    # acc_wn = cnt_wn / cnt
    # acc_wc = cnt_wc / cnt
    # acc_wo = cnt_wo / cnt
    # acc_wvi = cnt_wvi / cnt
    # acc_wv = cnt_wv / cnt
    # acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt
    # results_text2sql['logical form accuracy'] = acc_lx
    results_text2sql['execution accuracy'] = acc_x

    detailed_text2sql_acc = [acc_x]
    return results_slt, results_text2sql, all_text_pred, all_sql_query_pred, detailed_text2sql_acc, text2sql_mid_results, cnt_list


def print_result(print_data, dname):
    print(f'{dname} results ------------')
    print(print_data)

def print_detailed_text2sql_result(acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f"ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )

def save_text_pred_dev(save_root, all_text_pred, dname):
    with open(os.path.join(save_root, dname), 'w') as f:
        for text in all_text_pred:
            f.write(text+'\n')


if __name__ == '__main__':
    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    BERT_PT_PATH = '../annotated_wikisql_and_PyTorch_bert_param'
    path_sign2sql = '/home/huangwencan/data/sign2sql/dataset'

    path_save_for_evaluation = args.path_save_for_evaluation  #'../sign2text2sql_test_v2'
    if not os.path.exists(path_save_for_evaluation):
        os.mkdir(path_save_for_evaluation)
    
    path_pretrained_slt_model = '../sign2text_save'
    path_pretrained_text2sql_model = '../text2sql_save'
    
    ## 3. Load data

    test_data, table, test_loader = get_data(path_sign2sql, args)

    ## 4. Build & Load models
    
    if args.trained:
        # To start from the pre-trained models, un-comment following lines.
        path_model_bert = os.path.join(path_pretrained_text2sql_model, 'model_bert_best.pt')
        path_model_text2sql = os.path.join(path_pretrained_text2sql_model, 'model_best.pt')
        path_model_slt = os.path.join(path_pretrained_slt_model, 'model_best.pt')
        model_slt, model_text2sql, model_bert, tokenizer, bert_config = \
            get_models(args, BERT_PT_PATH, trained=True, 
            path_model_bert=path_model_bert, path_model_sign2text=path_model_slt, path_model_text2sql=path_model_text2sql)
    else:
        # error
        pass

    ## 5. Test
    with torch.no_grad():
        results_slt, results_text2sql, all_text_pred, all_sql_query_pred, detailed_text2sql_acc, text2sql_mid_results, cnt_list = \
            test(test_loader, table, model_slt, model_text2sql, model_bert, bert_config, tokenizer, 
                args.max_seq_length,
                args.num_target_layers, 
                detail=False, 
                st_pos=0, EG=args.EG, beam_size=4,
                path_db=path_sign2sql, db_name='all', separate_test=args.separate_test)

    print_result(results_slt, 'Sign to Text')
    print_result(results_text2sql, 'Text to SQL')
    # print_detailed_text2sql_result(detailed_text2sql_acc, 'text2sql detailed')
    
    save_text_pred_dev(path_save_for_evaluation, all_text_pred, 'sign2text_all_text_pred.txt')
    save_text_pred_dev(path_save_for_evaluation, all_sql_query_pred, 'text2sql_all_sql_query_pred.txt')
    save_for_evaluation(path_save_for_evaluation, text2sql_mid_results, 'text2sql_test')

