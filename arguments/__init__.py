#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.device="cuda:0"
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 1#-1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        
        self.scene_name="brandenburg"
        self.use_colors_precomp=True#True
        self.use_decode_with_pos=False
        

        self.use_indep_mask_branch=False
        
        # self.data_perturb=[]#for lego ["color","occ"]
        
        self.use_features_mask=True
        if self.use_features_mask:
            self.features_mask_loss_coef=0.15
            self.features_mask_iters=2500
        
        self.use_okmap=False #only use k feature maps
        self.use_kmap_pjmap=True # use k and projection feature maps
        
        #(num_prjection + K)feature maps 
        # K=2 or 3 is ok   
        if self.use_kmap_pjmap:
            self.map_num=1+2 
        elif self.use_okmap:
            self.map_num=2
        
        self.use_wo_adative=0  #without adaptive
        #init the coordinate with random weighted xyz
        self.use_xw_init_box_coord=True 
        self.use_color_net=True

        self.use_scaling_loss=False    
        self.use_lpips_loss=True
        
        self.use_box_coord_loss=True
        self.coord_scale=1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016*0.1
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.map_generator_lr=1e-3*2
        
        self.color_net_lr=5e-4
        
        self.box_coord_lr=1
        self.warm_up_iter=0
        
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002*2
        self.opacity_threshold=0.005
        self.random_background = False
        
        self.scaling_loss_coef=0.005#*0.2
        self.lpips_loss_coef=0.005
        self.box_coord_loss_coef=0.005*0.2
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)
    #args_cfgfile args_cmdline
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

