import torch
from dgl import function as fn
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
# from pnaconv import PNAConv

class Conv(nn.Module):
    r"""We modified existing implementation of GraphSAGE from DGL
    (https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/sageconv.py)
    """
    def __init__(self, in_feats, out_feats, norm=None, activation=None):
        super(Conv, self).__init__()
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.norm = norm
        self.activation = activation
        self.fc_pool_src = nn.Linear(in_feats, 4 * in_feats, bias=True)
        self.fc_pool_dst = nn.Linear(in_feats, 4 * in_feats, bias=True)
        self.fc_neigh = nn.Linear(in_feats + 4 * in_feats, out_feats, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_pool_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_pool_dst.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def forward(self, graph, feat):
        in_feats = self._in_feats
        graph = graph.local_var()
        graph.ndata['h_src'] = self.fc_pool_src(feat)
        
        graph.ndata['h_src_sum'], graph.ndata['h_src_mean'], graph.ndata['h_src_max'], graph.ndata['h_src_std'] = torch.split(graph.ndata.pop('h_src'), [in_feats, in_feats, in_feats, in_feats], dim=-1)
        graph.ndata['h_src_std2'] = (graph.ndata['h_src_std'] ** 2)
        graph.update_all(fn.u_mul_e('h_src_sum', 'weight', 'm_sum_src'), fn.sum('m_sum_src', 'neigh_sum_src'))
        graph.update_all(fn.u_mul_e('h_src_mean', 'weight', 'm_mean_src'), fn.mean('m_mean_src', 'neigh_mean_src'))
        graph.update_all(fn.u_mul_e('h_src_max', 'weight', 'm_max_src'), fn.max('m_max_src', 'neigh_max_src'))
        graph.update_all(fn.u_mul_e('h_src_std', 'weight', 'm_std1_src'), fn.mean('m_std1_src', 'neigh_std1_src'))
        graph.update_all(fn.u_mul_e('h_src_std2', 'weight', 'm_std2_src'), fn.mean('m_std2_src', 'neigh_std2_src'))
        graph.ndata['neigh_std_src'] = (graph.ndata['neigh_std2_src'] - (graph.ndata['neigh_std1_src'] ** 2))
        
        h_neigh = torch.cat((graph.ndata['neigh_sum_src'], graph.ndata['neigh_mean_src'], graph.ndata['neigh_max_src'], graph.ndata['neigh_std_src']), dim=-1)
        degs = graph.in_degrees()
        h_neigh[degs == 0, :] = 0
            
        rst = self.fc_neigh(torch.cat((feat, h_neigh), dim=1))
        
        if self.activation is not None:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        return rst

class MONSTOR(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers): # 4, 16, 3    
        super(MONSTOR, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        dims = [in_feats, *[n_hidden for _ in range(n_layers - 1)], 1] # [4, 16, 16, 1]

        for i in range(n_layers):
            self.layers.append(Conv(dims[i], dims[i+1])) # Conv(4, 16), Conv(16, 16), Conv(16, 1)
            self.acts.append(nn.ReLU())
        
    def forward(self, g, features):
        graph = g.local_var()
        h = features.clone()
        edge_feature = graph.edata['weight'].unsqueeze(1).cpu()
        for act, layer in zip(self.acts, self.layers): # (ReLU, Conv(4, 16))
            h = act(layer(graph, h)) # ReLU(Conv(4, 16)(graph, h))
        
        # compute upper bound of influence
        prv, now = features[:, -1]-features[:, -2], features[:, -1] 
        graph.ndata['prv_delta'] = prv
        lb = now
        
        graph.update_all(fn.u_mul_e('prv_delta', 'weight', 'm1'), fn.sum('m1', 'max_delta'))
        raw_result = lb + h.squeeze()
        clipped_result = torch.clamp(torch.min(torch.max(lb, raw_result), lb + graph.ndata['max_delta']), min=0., max=1.)
        if self.training: return raw_result
        else: return torch.clamp(raw_result, min=0., max=1.) # clipped_result
