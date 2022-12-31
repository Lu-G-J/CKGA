# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime
###################### changed
from transformers import BertModel
#from pytorch_pretrained_bert import BertModel
######################
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, random_split

from data_utils_bert import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import AFGCN, INTERGCN, AFGCN_BERT, INTERGCN_BERT

###################### changed
import sys
import pickle
sys.path.append('../')
from adapter_models import CONTROLER, ADAPTER, MHA
######################



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        ################################ changed (put the entity_idx into dataset)
        train_data = pickle.load(open( f'../graph/intergcn_extra_file/{opt.dataset}_train_{opt.adapter_score}_graphid.pkl','rb'))
        test_data = pickle.load(open( f'../graph/intergcn_extra_file/{opt.dataset}_test_{opt.adapter_score}_graphid.pkl','rb'))
        for i in range(len(self.trainset.data)):
            self.trainset.data[i]['entity_idx'] = train_data[i]
        for i in range(len(self.testset.data)):
            self.testset.data[i]['entity_idx'] = test_data[i]
        
        dataset_convert = {'lap14':'laptop', 'rest14':'restaurant', 'twitter':'twitter'}
        adapter = ADAPTER(f'../graph/{dataset_convert[opt.dataset]}', opt)
        mha = MHA(emb1_dim= opt.bert_dim, emb2_dim=opt.adapter_gcn_out_dim, hdim=opt.adapter_gcn_out_dim, n_head=4)
        if opt.fuse_mode == 'p':
            controler = CONTROLER(self.model, adapter, mha, opt.bert_dim, opt.polarities_dim)
        elif opt.fuse_mode =='c':
            controler = CONTROLER(self.model, adapter, mha, opt.bert_dim+opt.adapter_gcn_out_dim, opt.polarities_dim)
        self.model = controler
        self.model.to(opt.device)
        
        ################################ end 
        
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        #self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer1, optimizer2, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        ###################### changed
        deleted_path = 'init'
        deleted_arg_path = 'init'
        ######################
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            ###################### changed
            optimizer2.zero_grad()
            self.model.adapter(None, gcn=True)
            ######################
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer1.zero_grad()
                
                ######################################### changed (define the train method)
                #inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                origin_model_inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                adapter_inputs = batch['entity_idx'].to(self.opt.device)
                
                #outputs = self.model(inputs)
                outputs = self.model(origin_model_inputs, adapter_inputs, model=self.opt.adapter_mode, mode=self.opt.fuse_mode)
                targets = batch['polarity'].to(self.opt.device)
                ######################################### end
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)
                optimizer1.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                    val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
                    logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_f1_1 = val_f1
                        max_val_epoch = i_epoch
                        ###################### changed
                        if self.opt.save:
                            if not os.path.exists('state_dict/args'):
                                #os.mkdir('state_dict')
                                os.makedirs('state_dict/args')
                            '''
                            path = f"state_dict/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%val_acc}_f1_{'%.4f'%val_f1}.pkl"
                            arg_path = f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%val_acc}_f1_{'%.4f'%val_f1}_arg.pkl"

                            torch.save(self.model.state_dict(), path)
                            pickle.dump(self.opt, open(arg_path,'wb'))

                            deleted_path.append( f"state_dict/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%val_acc}_f1_{'%.4f'%val_f1}.pkl")
                            deleted_path.append( f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%val_acc}_f1_{'%.4f'%val_f1}_arg.pkl")
                            '''
                            if deleted_path != 'init' and deleted_arg_path != 'init':
                                try:
                                    os.remove(deleted_path)
                                    os.remove(deleted_arg_path)
                                except Exception as e:
                                    print(e)
                            deleted_path = f"state_dict/{self.opt.model_name}_{self.opt.dataset}_acc_{'%.4f'%val_acc}_f1_{'%.4f'%val_f1}.pkl"
                            deleted_arg_path =      f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_acc_{'%.4f'%val_acc}_f1_{'%.4f'%val_f1}_arg.pkl"
                            torch.save(self.model.state_dict(), deleted_path)
                            pickle.dump(self.opt, open(deleted_arg_path,'wb'))


                        ######################
                            logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
            ###################### changed
            optimizer2.step()
            ######################
        ########################
        '''
        for path in deleted_path:
            if path != f"state_dict/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%max_val_acc}_f1_{'%.4f'%max_val_f1_1}.pkl" and path != f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%max_val_acc}_f1_{'%.4f'%max_val_f1_1}_arg.pkl":
                os.remove(path)
        '''
        ########################
        
        return f"state_dict/{self.opt.model_name}_{self.opt.dataset}_acc_{'%.4f'%max_val_acc}_f1_{'%.4f'%max_val_f1_1}.pkl"

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                ######################################### changed (define the test method)
                #t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_origin_model_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_adapter_inputs = t_batch['entity_idx'].to(self.opt.device)
                t_targets = t_batch['polarity'].to(self.opt.device)
                #t_outputs = self.model(t_inputs)
                t_outputs = self.model(t_origin_model_inputs, t_adapter_inputs, model=self.opt.adapter_mode, mode=self.opt.fuse_mode)
                ######################################## end

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        ####################### changed
        if self.opt.train_model == 'd':
            self.model.origin_model.load_state_dict(torch.load(self.opt.origin_model_path))
            print(f'loaded checkpoint from {self.opt.origin_model_path}')
            
        
        params1 = [
            {"params": [p for p in self.model.origin_model.parameters() if p.requires_grad], "lr":self.opt.origin_model_lr},
            {"params": [p for p in self.model.classification.parameters() if p.requires_grad], "lr":self.opt.lr}, 
            {"params": [p for p in self.model.mha.parameters() if p.requires_grad], "lr":self.opt.lr}, 
        ]
        params2 = [
            {"params": [p for p in self.model.adapter.parameters() if p.requires_grad], "lr":self.opt.lr}
        ]
        optimizer1 = self.opt.optimizer(params1, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        optimizer2 = self.opt.optimizer(params2, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        
        
        ####################### end
        best_model_path = self._train(criterion, optimizer1, optimizer2, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='intergcn_bert', type=str)
    parser.add_argument('--dataset', default='rest14', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-4, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=30, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=30, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='/home/luguojun/transformers/bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    
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
    parser.add_argument('--adapter_gcn_out_dim', type=int, default=768)
    parser.add_argument('--adapter_dropout', type=float, default=0.5)
    parser.add_argument('--adapter_layer_num', type=int, default=2)
    parser.add_argument('--adapter_freeze_emb', type=str2bool, default=True)
    parser.add_argument('--adapter_mode', type=str, default='adapter',choices=['adapter', 'origin'])
    parser.add_argument('--adapter_kge', type=str, default='transh',choices=['transe', 'transh','transr','rotate'])
    parser.add_argument('--adapter_norm', type=str2bool, default='False')
    
    parser.add_argument('--fuse_mode', type=str, default='p', choices=['p','c'], help='plus or concatenate')
    parser.add_argument('--train_model', type=str, default='d', choices=['j','d'], help='joint or dependent')
    parser.add_argument('--origin_model_path', type=str, default='./best_origin_state_dict/intergcn_bert_rest14_val_acc_0.8643.pkl')
    parser.add_argument('--origin_model_lr', type=float, default=0)
    parser.add_argument('--save', type=str2bool, default='True')
    ###################################################################
    
    
    opt = parser.parse_args()

    if opt.seed is None:
        opt.seed = random.randint(0,99999999)
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'intergcn_bert': INTERGCN_BERT,
        'afgcn_bert': AFGCN_BERT,
    }
    dataset_files = {
        'rest14': {
            'train': './con_datasets/rest14_train.raw',
            'test': './con_datasets/rest14_test.raw'
            },
        'lap14': {
            'train': './con_datasets/lap14_train.raw',
            'test': './con_datasets/lap14_test.raw'
            },
        'rest15': {
            'train': './con_datasets/rest15_train.raw',
            'test': './con_datasets/rest15_test.raw'
            },
        'rest16': {
            'train': './con_datasets/rest16_train.raw',
            'test': './con_datasets/rest16_test.raw'
            },
    }
    input_colses = {
        'intergcn_bert': ['concat_bert_indices', 'concat_segments_indices', 'aspect_indices', 'left_indices', 'text_indices', 'dependency_graph', 'aspect_graph'],
        'afgcn_bert': ['concat_bert_indices', 'concat_segments_indices', 'aspect_indices', 'left_indices', 'text_indices', 'dependency_graph'],

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = './log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
