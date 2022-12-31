import torch
import math
import pickle
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from adapter_utils import obtain_adj_from_rdf
import torch.nn.functional as F

class CONTROLER(nn.Module):
    def __init__(self, origin_model, adapter, mha, hidden_dim, polarities_dim):
        super().__init__()
        self.origin_model = origin_model
        self.adapter = adapter
        self.classification = nn.Linear(hidden_dim, polarities_dim)
        self.mha = mha
        
    def forward(self,origin_inputs, adapter_inputs=None, model='adapter', mode='p',input_mode=None):
        '''
        model : origin/adapter
        '''
        if model == 'origin':
            if input_mode == 'dict':
                origin_output = self.origin_model(**origin_inputs)
            else:
                origin_output = self.origin_model(origin_inputs)
            return origin_output['classification']
        
        elif model == 'adapter':
            #origin_output = self.origin_model(origin_inputs)['emb']
            
            if input_mode == 'dict':
                origin_output = self.origin_model(**origin_inputs)
            else:
                origin_output = self.origin_model(origin_inputs)
            aspect_emb = origin_output['aspect_emb'].squeeze(-1) # [batch_size, hdim] 
            origin_output = origin_output['emb'] # [batch_size, hdim]
            
            
            adapter_output = self.adapter(adapter_inputs) # [batch_size, hdim]
            adapter_output = self.mha(aspect_emb, adapter_output) # [batch_size, hdim]
            
            
            if mode == 'p':
                #print(origin_output.size(), adapter_output.size())
                hidden_emb = origin_output + adapter_output
            elif mode == 'c':
                hidden_emb = torch.cat((origin_output,adapter_output), dim=-1)
                
            classification = self.classification(hidden_emb)
            return classification
         
    def freeze_origin_model(self,):
        for parameter in self.origin_model.parameters():
            parameter.requires_grad=False         

class ADAPTER(nn.Module):
    def __init__(self, data_path, args):
        super().__init__()
        ent_kge_matrix = pickle.load(open(f'{data_path}_kge_{args.adapter_kge}.pkl', 'rb'))
        ent_abs_bert_matrix = pickle.load(open(data_path+'_abs_bert.pkl', 'rb'))
        
        '''
        1. prepare the KG embedding
        2. append last idx <unk>, using zeros vector
        '''
        ent_kge_matrix = np.vstack((ent_kge_matrix, np.zeros(ent_kge_matrix.shape[-1])))
        ent_abs_bert_matrix = np.vstack((ent_abs_bert_matrix, np.zeros(ent_abs_bert_matrix.shape[-1])))
        
        '''if choosing to freeze emb, then set False to requires_grad'''
        '''
        self.ent_kge = Variable(torch.tensor(ent_kge_matrix, dtype=torch.float32), requires_grad=not args.adapter_freeze_emb).to(args.device)
        self.ent_abs_bert = Variable(torch.tensor(ent_abs_bert_matrix, dtype=torch.float32), requires_grad=not args.adapter_freeze_emb).to(args.device)
        
        '''
        self.ent_kge = nn.Embedding.from_pretrained(torch.tensor(ent_kge_matrix, dtype=torch.float32),freeze=args.adapter_freeze_emb)
        self.ent_abs_bert = nn.Embedding.from_pretrained(torch.tensor(ent_abs_bert_matrix, dtype=torch.float32),freeze=args.adapter_freeze_emb)
        self.ent_idx = torch.arange(0,self.ent_kge.weight.size()[0]).to(args.device)
        
        ######### normalize
        self.norm = False
        if args.adapter_norm is True:
            self.ent_kge_norm = nn.LayerNorm(self.ent_kge.weight.size())
            self.ent_abs_bert_norm = nn.LayerNorm(self.ent_abs_bert.weight.size())
            self.norm = True
        
        '''
        1. prepare the KG by rdf
        2. set the GCN with KG
        '''
        temp = pickle.load(open(data_path+'_graph.pkl','rb'))
        rdfs = np.array(temp['RDFs'])
        #rdfs = np.array(list(set([(i[0],i[1]) for i in temp['RDFs']])))
        
        
        entityTotal = len(temp['e2GraphID']) + 1 # add <unk>
        adj = obtain_adj_from_rdf(rdfs, entityTotal)
        adj = adj.to(args.device)

        
        self.GCN = GCN(
            nfeat = self.ent_kge.weight.size()[-1] + self.ent_abs_bert.weight.size()[-1],
            nhid = args.adapter_gcn_hid_dim,
            adj = adj,
            out = args.adapter_gcn_out_dim,
            dropout = args.adapter_dropout,
            layer_num = args.adapter_layer_num   
        )

    def forward(self, entities_idx, gcn=False):
        if gcn is False:
            return self.ent_emb_output[entities_idx]   
        
        ent_kge = self.ent_kge(self.ent_idx) # [ent_num, kge_dim]
        ent_abs_bert = self.ent_abs_bert(self.ent_idx) #[ent_num, abs_dim]

        if self.norm:
            ent_emb_input = torch.cat((
                self.ent_kge_norm(ent_kge), 
                self.ent_abs_bert_norm(ent_abs_bert)),
                dim=-1)
        else:
            ent_emb_input = torch.cat((ent_kge, ent_abs_bert),dim=-1) # [ent_num, kge_dim+abs_dim]
        #
        self.ent_emb_output = self.GCN(ent_emb_input) # [ent_num, gcn_out_dim]
        return None

        
                 
            
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout, layer_num, adj=None):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        if adj is not None:
            self.adj = adj
        layers = []
        if layer_num == 1:
            layers.append(GraphConvolution(nfeat, out))
        else:
            layers.append(GraphConvolution(nfeat, nhid))
            for _ in range(layer_num-2):
                layers.append(GraphConvolution(nhid, nhid))
            layers.append(GraphConvolution(nhid, out))
        self.layers = nn.ModuleList(layers)
            
    def forward(self, x, adj=None):
        if adj is None:
            adj = self.adj
        
        for i in range(self.layer_num):
            x = F.relu(self.layers[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MHA(nn.Module):
    def __init__(self, emb1_dim, emb2_dim, hdim, n_head):
        super().__init__()
        self.Q1 = nn.Linear(emb1_dim, hdim)
        self.K2 = nn.Linear(emb2_dim, hdim)
        self.V2 = nn.Linear(emb2_dim, hdim)
        self.n_head = n_head
        self.hdim = hdim

    def forward(self, emb1, emb2):
        batch_size = emb1.size()[0]
        emb1_Q = self.Q1(emb1).view(batch_size, -1, self.n_head, self.hdim//self.n_head).permute(0, 2, 1, 3) # [batch_size, n_heads, 1, hdim//n_head]
        emb2_K = self.K2(emb2).view(batch_size, -1, self.n_head, self.hdim//self.n_head).permute(0, 2, 1, 3)
        emb2_V = self.V2(emb2).view(batch_size, -1, self.n_head, self.hdim//self.n_head).permute(0, 2, 1, 3) 

        emb2_att = torch.matmul(emb1_Q,emb2_K.permute(0, 1, 3, 2)) # [batch_size, n_heads, 1, 1]
        emb2_out = torch.sigmoid(emb2_att) * emb2_V # [batch_size, n_heads, 1, hdim//n_head]  

        emb2_out = emb2_out.permute(0, 2, 1, 3).contiguous() # [batch_size, 1, n_heads, hdim//n_head]
        emb2_out = emb2_out.view(batch_size, -1, self.hdim) # [batch_size, 1, h_dim]

        return emb2_out.squeeze(1) # [batch_size, h_dim]