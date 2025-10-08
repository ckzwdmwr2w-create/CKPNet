
from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
from typing import Callable
import torch.nn as nn
import torch
import torch.nn.functional as F

#import matplotlib.pyplot as plt


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.W1 = torch.nn.Linear(768, 768, bias= True)
        self.W2 = torch.nn.Linear(768, 1, bias= True)
        self.tanh = torch.nn.Tanh()

    def forward(self, imput_tensor):
        A  = self.W2(self.tanh(self.W1(imput_tensor)))
        output_tensor = torch.matmul(A.T, imput_tensor)  # torch.Size([1, 768])

        return output_tensor



class NodeNetwork(nn.Module):
    def __init__(self, in_features):
        super(NodeNetwork, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(self.in_features, self.in_features)
        self.layernorm = nn.LayerNorm(self.in_features)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
        self.net = nn.Sequential(
                       nn.Linear(2 * self.in_features, self.in_features),
                       nn.Dropout(0.1),
                       nn.LayerNorm(self.in_features),
                       nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x, residual):
        x = self.net(x)
        out = self.fc(x)
        out = self.layernorm(out)
        out += residual
        out = self.relu(out)
        out = self.fc(out)
        
        return out


class Sim_EdgeNetwork(nn.Module):
    def __init__(self, in_features):
        super(Sim_EdgeNetwork, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(self.in_features, self.in_features)
        self.LN = nn.LayerNorm(self.in_features)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
        self.fc_map = nn.Linear(self.in_features, 1)
        self.net = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.LayerNorm(self.in_features),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.arithm_sim = torch.sub

    def residual_net(self, x, residual):
        out = self.fc(x)
        out = self.LN(out)

        out += residual
        out = self.relu(out)

        return out
    
    def network(self, x_ij, x_i, x_j):
        out = self.net(x_ij)
        res = self.arithm_sim(x_i, x_j)   
        out = self.residual_net(out, res)  
        out = self.fc_map(out)
        out = torch.transpose(out, 0, 2)
        return out

    def forward(self, node_feat):
        '''
        node_feat:  torch.Size([25, 768])
        '''
        x_i = node_feat.unsqueeze(1)   # torch.Size([25, 1, 768])
        x_j = torch.transpose(x_i, 0, 1)  # torch.Size([1, 25, 768]) 
        x_ij = torch.min(x_i, x_j)   # torch.Size([25, 25, 768]) 
        out = self.network(x_ij, x_i, x_j)   #[1, 25, 25]
        #print(out.size())
        return out


class NodeUpdateNetwork(nn.Module):
    def __init__(self, args):
        super(NodeUpdateNetwork, self).__init__()
        self.device         = args.device
        self.NodeNetwork    = NodeNetwork(in_features = 768)
        


    def forward(self, node_feat, edge_feat):
    
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        # get eye matrix 
        diag_mask = 1.0 - torch.eye(node_feat.size(0)).to(self.device)
        # 忽视自己跟自己的特征
        edge_feat = F.normalize(edge_feat * diag_mask)
        aggr_feat = torch.bmm(edge_feat.unsqueeze(0), node_feat.unsqueeze(0)).squeeze(0)
        #print('node_feat: ', node_feat.size())
        #print('aggr_feat: ', aggr_feat.size())
        #exit()
        ext_node_feat = torch.cat([node_feat, aggr_feat], -1)    #  torch.Size([25, 768*2])
        updated_node_feat = self.NodeNetwork(x = ext_node_feat, residual = node_feat)
        
        return updated_node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, device):
        super(EdgeUpdateNetwork, self).__init__()
        self.device = device
        self.sim_network = Sim_EdgeNetwork(in_features = 768)
        self.hidden_dim     = 768
        self.W_k = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.W_q = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, node_feat):
        K = self.W_k(node_feat)  
        Q = self.W_q(node_feat)
        edge_feat = F.normalize(torch.matmul(Q, K.transpose(0, -1)), -1)  # torch.Size([16, 21, 21])
        return edge_feat

class GraphNetwork(nn.Module):
    def __init__(self, args):
        super(GraphNetwork, self).__init__()

        self.device = args.device
        # set node to edge
        node2edge_net = EdgeUpdateNetwork(self.device)
        # set edge to node
        edge2node_net = NodeUpdateNetwork(args)

        

        self.add_module('node2edge_net{}'.format(0), node2edge_net)
        self.add_module('edge2node_net{}'.format(0), edge2node_net)
        


    # forward
    def forward(self, node_feat):    
        edge_feat = self._modules['node2edge_net{}'.format(0)](node_feat)
        node_feat = self._modules['edge2node_net{}'.format(0)](node_feat, edge_feat)
        
        return node_feat, edge_feat







   
     


