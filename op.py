
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import sys


class SetopResBasicBlock(nn.Module):
    """A basic setops residual layer.

    Applies Linear+BN on the input x and adds residual. Applies leaky-relu on top.
    """
    def __init__(self, latent_dim):
        super(SetopResBasicBlock, self).__init__()

        self.fc = nn.Linear(latent_dim, latent_dim)
        self.layernorm = torch.nn.LayerNorm(latent_dim)

        self.relu = nn.LeakyReLU(0.3, inplace=True)

        #self.relu = nn.ReLU()
        #self.relu = nn.Tanh()


    def forward(self, x, residual):

        out = self.fc(x)
        out = self.layernorm(out)

        out += residual
        out = self.relu(out)

        return out


class SetopResBlock(nn.Module):
    
    def __init__(
            self,
            input_dim: int,
            layers_num: int,
            arithm_op: Callable,
            **kwargs):

        super(SetopResBlock, self).__init__()


        self.net_ab = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(0.3, inplace=True),
        )

        self.res_layers = []
        for i in range(layers_num):
            layer_name = "res_layer{}".format(i)
            setattr(self, layer_name, SetopResBasicBlock(input_dim))
            self.res_layers.append(layer_name)

        self.fc_out = nn.Linear(input_dim, input_dim)
        self.arithm_op = arithm_op

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        out = torch.cat((a, b), dim=-1)
        out = self.net_ab(out)

        res = self.arithm_op(a, b)

        for layer_name in self.res_layers:
            layer = getattr(self, layer_name)
            out = layer(out, res)

        out = self.fc_out(out)

        return out

#SetopResBlock_v2 比 SetopResBlock_v1 多了一个dropout_ratio

def subrelu(x, y):
    return F.relu(x-y)


class SubtractOp(nn.Module):
    def __init__(self, input_dim, S_layers_num):
        super(SubtractOp, self).__init__()
        self.basic_block_cls    = 'SetopResBasicBlock'  #block_cls 里面的核心模块
        self.input_dim          = input_dim
        self.S_layers_num       = S_layers_num
        self.arithm_op          = torch.sub
        #self.arithm_op          = subrelu
        self.block_cls          = SetopResBlock(self.input_dim, self.S_layers_num, self.arithm_op)
    
    def forward(self, a, b):

        a_S_b = self.block_cls(a, b)
        #b_S_a = self.block_cls(b, a)
        
        #a_S_b_b = self.block_cls(a_S_b, b)
        #b_S_a_a = self.block_cls(b_S_a, a)

        '''
        a_S_b_I_a = self.block_cls(a, b_I_a)
        b_S_a_I_b = self.block_cls(b, a_I_b)
        a_S_a_I_b = self.block_cls(a, a_I_b)
        b_S_b_I_a = self.block_cls(b, b_I_a)
        '''

        return a_S_b

        
class IntersectOp(nn.Module):
    def __init__(self, input_dim, I_layers_num):
        super(IntersectOp, self).__init__()
        self.basic_block_cls    = 'SetopResBasicBlock'  #block_cls 里面的核心模块
        self.input_dim          = input_dim
        self.I_layers_num       = I_layers_num
        #self.arithm_op          = torch.min
        self.arithm_op          = torch.mul
        self.block_cls          = SetopResBlock(self.input_dim, self.I_layers_num, self.arithm_op)
    
    def forward(self, a, b):

        a_I_b = self.block_cls(a, b)
        b_I_a = self.block_cls(b, a)

        a_I_b_b = self.block_cls(a_I_b, b)
        b_I_a_a = self.block_cls(b_I_a, a)

        return a_I_b, b_I_a, a_I_b_b, b_I_a_a


class UnionOp(nn.Module):
    def __init__(self, input_dim, U_layers_num):
        super(UnionOp, self).__init__()
        self.basic_block_cls    = 'SetopResBasicBlock'  #block_cls 里面的核心模块
        self.input_dim          = input_dim
        self.U_layers_num       = U_layers_num
        #self.arithm_op          = torch.max
        self.arithm_op          = torch.add
        self.block_cls          = SetopResBlock(self.input_dim, self.U_layers_num, self.arithm_op)

    def forward(self, a, b):
        a_U_b = self.block_cls(a, b)
        b_U_a = self.block_cls(b, a)

        #a_U_b_b = self.block_cls(a_U_b, b)
        #b_U_a_a = self.block_cls(b_U_a, a)

        '''
        out_a = self.union_op(a_S_b_I_a, a_I_b)
        out_b = self.union_op(b_S_a_I_b, b_I_a)
        '''

        return a_U_b, b_U_a


if __name__ == '__main__':
    a = torch.tensor([[1.,3,5,7,9], [1.,3,5,7,9]])
    b = torch.tensor([[2,4.,6,8,10], [1.,3,5,7,9]])

    inte = IntersectOp(input_dim = 5, I_layers_num = 1)

    a_I_b, b_I_a, a_I_b_b, b_I_a_a = inte(a, b)
    print(a_I_b)

    '''
    sub = SubtractOp(input_dim = 5, S_layers_num = 1)
    a_S_b, b_S_a, a_S_b_b, b_S_a_a = sub(a,b)
    print(a_S_b, b_S_a, a_S_b_b, b_S_a_a)
    print()

    

    union = UnionOp(input_dim = 5, U_layers_num = 1)
    a_U_b, b_U_a, a_U_b_b, b_U_a_a = union(a, b)

    print(a_U_b, b_U_a, a_U_b_b, b_U_a_a)

    print(torch.cat((a, b), 0))

    '''