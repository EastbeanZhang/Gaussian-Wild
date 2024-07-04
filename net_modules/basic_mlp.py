import torch
import torch.nn as nn
from net_modules.embedder import *
import numpy as np

class lin_module(nn.Module):
    def __init__(self,
                d_in,
                d_out,
                dims,
                multires=0,
                act_fun=None,last_act_fun=None,weight_norm=False,weight_zero=False,weight_xavier=True):
        super().__init__()
        
        dims = [d_in] + dims + [d_out]
        self.num_layers = len(dims)
        if act_fun==None:
            self.act_fun = nn.Softplus(beta=100)
        else:
            self.act_fun=act_fun
        self.last_act_fun=last_act_fun
        
        for l in range(0, self.num_layers -1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_zero:
                torch.nn.init.normal_(lin.weight, 0.0,0.0)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            if weight_xavier:
                torch.nn.init.xavier_normal_(lin.weight)
                torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

    def forward(self, inx):

        x = inx
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            
            if l==self.num_layers-2:
                if self.last_act_fun is not None:
                    x=self.last_act_fun(x)
            else:
                x = self.act_fun(x)
        return x



        

        
