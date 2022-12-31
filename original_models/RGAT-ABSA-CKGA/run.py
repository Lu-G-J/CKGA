# coding=utf-8
import argparse
import logging
import os

import random

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import (Aspect_Text_GAT_ours,
                    Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)
from trainer import train


###################### changed
import sys
import pickle
sys.path.append('../')
from adapter_models import CONTROLER, ADAPTER, MHA
######################

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args(jupyter=False):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='laptop',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--output_dir', type=str, default='./data/output-gcn-laptop',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')


    parser.add_argument('--cuda_id', type=str, default='0',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--glove_dir', type=str, default='/data1/SHENWZH/wordvec',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='/home/luguojun/transformers/bert-base-uncased',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')

    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')


    parser.add_argument('--add_non_connect',  type= bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type= bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type = int, default=4,
                        help='max number of hops')


    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')


    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    # GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type = str, choices=['linear','dotprod','gcn'], default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--embedding_type', type=str,default='glove', choices=['glove','bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    
    ###################################################################
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser.add_argument('--adapter_gcn_hid_dim', type=int, default=300)
    parser.add_argument('--adapter_score', type=int, default=0, choices=[i for i in range(-10,12,2)])
    parser.add_argument('--adapter_gcn_out_dim', type=int, default=300)
    parser.add_argument('--adapter_dropout', type=float, default=0.5)
    parser.add_argument('--adapter_layer_num', type=int, default=2)
    parser.add_argument('--adapter_freeze_emb', type=str2bool, default=True)
    parser.add_argument('--adapter_mode', type=str, default='adapter',choices=['adapter', 'origin'])
    parser.add_argument('--adapter_kge', type=str, default='transh',choices=['transe', 'transh','transr','rotate'])
    parser.add_argument('--adapter_norm', type=str2bool, default='False')
    
    parser.add_argument('--fuse_mode', type=str, default='p', choices=['p','c'], help='plus or concatenate')
    parser.add_argument('--train_model', type=str, default='d', choices=['j','d'], help='joint or dependent')
    parser.add_argument('--origin_model_path', type=str, default='./best_origin_state_dict/laptop_acc_0.7665_f1_0.7202.pkl')
    parser.add_argument('--origin_model_lr', type=float, default=0)
    
    ###################################################################
    
    
    
    if jupyter is True:
        return parser.parse_args(args=[])
    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))
        


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    # Parse args
    args = parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    #print(os.environ["CUDA_VISIBLE_DEVICES"])
    check_args(args)

    # Setup CUDA, GPU training
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    if args.seed is None:
        args.seed = random.randint(0,99999999)
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer

    # Load datasets and vocabs
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab= load_datasets_and_vocabs(args)

    ################################ changed
    train_data = pickle.load(open(f'../graph/rgat_extra_file/{args.dataset_name}_train_{args.adapter_score}_graphid.pkl','rb'))
    test_data = pickle.load(open(f'../graph/rgat_extra_file/{args.dataset_name}_test_{args.adapter_score}_graphid.pkl','rb'))

    for idx, data in enumerate(train_data):
        train_dataset.data[idx]['entity_idx'] = torch.tensor(train_data[idx])

    for idx, data in enumerate(test_data):
        test_dataset.data[idx]['entity_idx'] = torch.tensor(test_data[idx])
    
    ################################ end
    
    
    # Build Model
    # model = Aspect_Text_Multi_Syntax_Encoding(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    if args.pure_bert:
        model = Pure_Bert(args)
    elif args.gat_bert:
        model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert
    elif args.gat_our:
        model = Aspect_Text_GAT_ours(args, dep_tag_vocab['len'], pos_tag_vocab['len']) # R-GAT with reshaped tree
    else:
        model = Aspect_Text_GAT_only(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # original GAT with reshaped tree

    ######################################### changed (define the controler with origin_model and adapter)
    dataset_convert = {'laptop':'laptop', 'rest':'restaurant', 'twitter':'twitter'}
    adapter = ADAPTER(f'../graph/{dataset_convert[args.dataset_name]}', args)
    if args.gat_bert:
        mha = MHA(emb1_dim=args.embedding_dim, emb2_dim=args.adapter_gcn_out_dim, hdim=args.adapter_gcn_out_dim, n_head=4)
    else:
        mha = MHA(emb1_dim=args.hidden_size*2, emb2_dim=args.adapter_gcn_out_dim, hdim=args.adapter_gcn_out_dim, n_head=4)
    origin_model = model
    if args.fuse_mode == 'p':
        controler = CONTROLER(origin_model, adapter, mha, args.final_hidden_size, args.num_classes)
    elif args.fuse_mode =='c':
        controler = CONTROLER(origin_model, adapter, mha, args.final_hidden_size+args.adapter_gcn_out_dim, args.num_classes)
    model = controler
    #print(model)
    ######################################## end
    model.to(args.device)
    # Train
    _, _,  all_eval_results = train(args, train_dataset, model, test_dataset)

    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc']) 
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))


if __name__ == "__main__":
    main()

