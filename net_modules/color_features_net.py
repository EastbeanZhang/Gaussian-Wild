import torch
import torch.nn as nn
from net_modules.embedder import *
from net_modules.basic_mlp import lin_module


class Color_net(nn.Module):
    def __init__(self,
                fin_dim,
                pin_dim,
                view_dim,
                pfin_dim,
                en_dims,
                de_dims,
                multires,
                pre_compc=False,
                cde_dims=None,
                use_pencoding=[False,False],#postion viewdir
                weight_norm=False,
                weight_xavier=True,
                use_drop_out=False,
                use_decode_with_pos=False,
                ):
        super().__init__()
        self.pre_compc=pre_compc
        self.use_pencoding=use_pencoding
        self.embed_fns=[]
        self.cache_outd=None
        self.use_decode_with_pos=use_decode_with_pos
        if use_pencoding[0]:
            embed_fn, input_ch = get_embedder(multires[0])
            pin_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)
            
        if use_pencoding[1]:
            embed_fn, input_ch = get_embedder(multires[1])
            view_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)

            
        self.encoder=lin_module(fin_dim+pin_dim+pfin_dim,fin_dim,en_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        self.decoder=lin_module(fin_dim*2,fin_dim,de_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        if self.pre_compc:
            # view_dim=3
            self.color_decoder=lin_module(fin_dim+view_dim,3,cde_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
            #self.color_decoder=lin_module(fin_dim+pin_dim,3,cde_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        self.use_drop_out=use_drop_out
        
        if use_drop_out:
            self.drop_outs=[nn.Dropout(0.1)]

                
    def forward(self, inp,inf,inpf,view_direction=None,inter_weight=1.0,store_cache=False):
        oinp=inp
        if self.use_drop_out:
            inpf=self.drop_outs[0](inpf)
        if  self.use_pencoding[0]:
            if self.use_decode_with_pos:
                oinp=inp.clone()
            inp = self.embed_fns[0](inp)

        if  self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
            #view_direction=self.embed_fn(view_direction)
        p_num=inf.shape[0]
        inf=inf.reshape([p_num,-1])
        
        inpf=inpf*inter_weight
        inx=torch.cat([inp,inpf,inf],dim=1)
        #inx=torch.cat([inp,inf],dim=1)
        oute= self.encoder(inx)
        outd=self.decoder(torch.cat([oute,inf],dim=1))
        if store_cache:
            self.cache_outd=outd
        else:
            self.cache_outd=None
            
        if self.pre_compc:
            if self.use_decode_with_pos:
                outc=self.color_decoder(torch.cat([outd,oinp],dim=1))
            else:
                outc=self.color_decoder(torch.cat([outd,view_direction],dim=1)) #view_direction
            return outc
        return outd.reshape([p_num,-1,3])

    def forward_cache(self, inp,view_direction=None):
        oinp=inp
        if  self.use_pencoding[0]:
            if self.use_decode_with_pos:
                oinp=inp.clone()
            inp = self.embed_fns[0](inp)
        if  self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
        p_num=inp.shape[0]
        if self.pre_compc:
            if self.use_decode_with_pos:
                outc=self.color_decoder(torch.cat([self.cache_outd,oinp],dim=1))
            else:
                outc=self.color_decoder(torch.cat([self.cache_outd,view_direction],dim=1)) #view_direction
            return outc
        return self.cache_outd.reshape([p_num,-1,3])
        

    

        