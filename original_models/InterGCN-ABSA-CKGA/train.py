# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import os
import math
import pickle
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import AFGCN, INTERGCN

###################### changed
import sys
import pickle
sys.path.append('../')
from adapter_models import CONTROLER, ADAPTER, MHA
######################


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.absa_dataset = absa_dataset
        ######################################### changed (put the entity_idx into dataset)
        train_data = pickle.load(open(f'../graph/intergcn_extra_file/{opt.dataset}_train_{opt.adapter_score}_graphid.pkl','rb'))
        test_data = pickle.load(open(f'../graph/intergcn_extra_file/{opt.dataset}_test_{opt.adapter_score}_graphid.pkl','rb'))
        for i in range(len(absa_dataset.train_data.data)):
            absa_dataset.train_data.data[i]['entity_idx'] = train_data[i]
        for i in range(len(absa_dataset.test_data.data)):
            absa_dataset.test_data.data[i]['entity_idx'] = test_data[i]  
        ########################################## end
        
        
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        
        
        ######################################### changed (define the controler with origin_model and adapter)
        dataset_convert = {'lap14':'laptop', 'rest14':'restaurant', 'twitter':'twitter'}
        #self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        origin_model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        
        adapter = ADAPTER(f'../graph/{dataset_convert[opt.dataset]}', opt)
        mha = MHA(emb1_dim=2*opt.hidden_dim, emb2_dim=opt.adapter_gcn_out_dim, hdim=opt.adapter_gcn_out_dim, n_head=4)
        if opt.fuse_mode == 'p':
            controler = CONTROLER(origin_model, adapter, mha, 2*opt.hidden_dim, opt.polarities_dim)
        elif opt.fuse_mode =='c':
            controler = CONTROLER(origin_model, adapter, mha, 2*opt.hidden_dim+opt.adapter_gcn_out_dim, opt.polarities_dim)
        self.model = controler
        self.model.to(opt.device)
        #print(self.model)
        ########################################## end
        
        
        #self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer1, optimizer2):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        
        #########################
        deleted_path = 'init'
        deleted_arg_path = 'init'
        #########################
        
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            
            ###################### changed
            self.model.train()
            optimizer2.zero_grad()
            self.model.adapter(None, gcn=True)
            ######################
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                self.model.train()
                optimizer1.zero_grad()
                #optimizer2.zero_grad()
                ######################################### changed (define the train method)   
                #inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                origin_model_inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                adapter_inputs = sample_batched['entity_idx'].to(self.opt.device)
                targets = sample_batched['polarity'].to(self.opt.device)

                #outputs = self.model(inputs)
                #self.model.adapter(None, gcn=True)
                outputs = self.model(origin_model_inputs, adapter_inputs, model=self.opt.adapter_mode, mode=self.opt.fuse_mode)
                ######################################### end
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)
                optimizer1.step()
                #optimizer2.step()
                
                

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        max_test_acc_1 = test_acc
                        
                        if not os.path.exists('state_dict/args'):
                            os.makedirs('state_dict/args')
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            ###################### changed
                            '''
                            torch.save(self.model.state_dict(), f"state_dict/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%test_acc}_f1_{'%.4f'%test_f1}.pkl")
                            pickle.dump(self.opt, open(f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%test_acc}_f1_{'%.4f'%test_f1}_arg.pkl",'wb'))
                            deleted_path.append(f"state_dict/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%test_acc}_f1_{'%.4f'%test_f1}.pkl")
                            deleted_path.append(f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%test_acc}_f1_{'%.4f'%test_f1}_arg.pkl")
                            '''
                            if deleted_path != 'init' and deleted_arg_path != 'init':
                                try:
                                    os.remove(deleted_path)
                                    os.remove(deleted_arg_path)
                                except Exception as e:
                                    print(e)
                            deleted_path = f"state_dict/{self.opt.model_name}_{self.opt.dataset}_acc_{'%.4f'%test_acc}_f1_{'%.4f'%test_f1}.pkl"
                            deleted_arg_path =      f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_acc_{'%.4f'%test_acc}_f1_{'%.4f'%test_f1}_arg.pkl"
                            torch.save(self.model.state_dict(), deleted_path)
                            pickle.dump(self.opt, open(deleted_arg_path,'wb'))
                            
                            ######################
                            
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))
            ###################### changed
            optimizer2.step() # update adapter
            ######################
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    break
            else:
                continue_not_increase = 0    
        ########################
        '''
        for path in deleted_path:
            if path != f"state_dict/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%max_test_acc_1}_f1_{'%.4f'%max_test_f1}.pkl" and path != f"state_dict/args/{self.opt.model_name}_{self.opt.dataset}_{self.opt.adapter_kge}_{self.opt.train_model}_{self.opt.origin_model_lr}_acc_{'%.4f'%max_test_acc_1}_f1_{'%.4f'%max_test_f1}_arg.pkl":
                os.remove(path)
        '''
        ########################
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                ######################################### changed (define the test method)
                #t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_origin_model_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_adapter_inputs = t_sample_batched['entity_idx'].to(self.opt.device)
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                #t_outputs = self.model(t_inputs)
                t_outputs = self.model(t_origin_model_inputs, t_adapter_inputs, model=self.opt.adapter_mode, mode=self.opt.fuse_mode)
                ######################################### end
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat: ', (i+1))
            self._reset_params()
            ####################### changed
            if self.opt.train_model == 'd':
                self.model.origin_model.load_state_dict(torch.load(self.opt.origin_model_path))
                print(f'loaded checkpoint from {self.opt.origin_model_path}')
                
                # new
            params1 = [
                {"params": [p for p in self.model.origin_model.parameters() if p.requires_grad], "lr":self.opt.origin_model_lr},
                {"params": [p for p in self.model.classification.parameters() if p.requires_grad], "lr":self.opt.learning_rate}, 
                {"params": [p for p in self.model.mha.parameters() if p.requires_grad], "lr":self.opt.learning_rate}, 
            ]
            params2 = [
                {"params": [p for p in self.model.adapter.parameters() if p.requires_grad], "lr":self.opt.learning_rate}
            ]
            optimizer1 = self.opt.optimizer(params1, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            optimizer2 = self.opt.optimizer(params2, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
                
            ####################### end
            max_test_acc, max_test_f1 = self._train(criterion, optimizer1, optimizer2)
            if max_test_acc > max_test_acc_avg:
                max_test_acc_avg = max_test_acc
            if max_test_f1 > max_test_f1_avg:
                max_test_f1_avg = max_test_f1
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg)
        print("max_test_f1_avg:", max_test_f1_avg)
        f_out.write('max_test_acc_avg: {0}, max_test_f1_avg: {1}'.format(max_test_acc_avg, max_test_f1_avg))

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='intergcn', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=30, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default=None, type=str)
    
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
    parser.add_argument('--save', default=True, type=str2bool)
    
    parser.add_argument('--adapter_gcn_hid_dim', type=int, default=300)
    parser.add_argument('--adapter_score', type=int, default=0, choices=[i for i in range(-10,12,2)])
    parser.add_argument('--adapter_gcn_out_dim', type=int, default=600)
    parser.add_argument('--adapter_dropout', type=float, default=0.5)
    parser.add_argument('--adapter_layer_num', type=int, default=2)
    parser.add_argument('--adapter_freeze_emb', type=str2bool, default=True)
    parser.add_argument('--adapter_mode', type=str, default='adapter',choices=['adapter', 'origin'])
    parser.add_argument('--adapter_kge', type=str, default='transh',choices=['transe', 'transh','transr','rotate'])
    parser.add_argument('--adapter_norm', type=str2bool, default='False')
    
    parser.add_argument('--fuse_mode', type=str, default='p', choices=['p','c'], help='plus or concatenate')
    parser.add_argument('--train_model', type=str, default='d', choices=['j','d'], help='joint or dependent')
    parser.add_argument('--origin_model_path', type=str, default='./best_origin_state_dict/intergcn_lap14_acc_0.6082.pkl')
    parser.add_argument('--origin_model_lr', type=float, default=0)
    ###################################################################
    
    
    opt = parser.parse_args()

    model_classes = {
        'afgcn': AFGCN,
        'intergcn': INTERGCN,
    }
    input_colses = {
        'afgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'intergcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'aspect_graph'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
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
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    
    if opt.seed is None:
        opt.seed = random.randint(0,999999999)
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
