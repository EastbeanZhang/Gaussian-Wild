

from argparse import ArgumentParser, Namespace
import sys
import os


def argument_init(args):
    
    args.data_perturb=args.data_perturb#["color","occ"] 
    if args.use_okmap:
            args.map_num=args.map_num#3
            args.map_generator_type="unet"
            args.feature_maps_dim=16
            args.feature_maps_combine="cat"
            args.use_indep_box_coord=True
            
            args.use_xw_init_box_coord=args.use_xw_init_box_coord
            if args.map_generator_type in ["unet"]:
                 args.map_generator_params={
                    "features_dim":args.feature_maps_dim*args.map_num,
                    "backbone":"resnet18",
                    "use_features_mask":args.use_features_mask,
                    "use_independent_mask_branch":args.use_indep_mask_branch
                 }
            
    elif args.use_kmap_pjmap:
            args.map_num=args.map_num
            args.map_generator_type="unet"
            args.feature_maps_dim=16
            args.feature_maps_combine="cat"
            args.use_indep_box_coord=True
            args.use_wo_adative=args.use_wo_adative
            args.use_xw_init_box_coord=args.use_xw_init_box_coord

            if  args.map_generator_type=="unet":
                args.map_generator_params={
                    "features_dim":args.feature_maps_dim*args.map_num,
                    "backbone":"resnet18",
                    "use_features_mask":args.use_features_mask,
                    "use_independent_mask_branch":args.use_indep_mask_branch
                }
            else:
                raise NotImplementedError
                                     
    if args.use_okmap:
        args.features_dim=args.feature_maps_dim*args.map_num
    elif args.use_kmap_pjmap: 
        args.features_dim=args.feature_maps_dim*args.map_num
    else: 
        args.features_dim=0
            
    args.use_color_net=True
    if args.use_color_net:
            args.color_net_type="naive"
            args.features_weight_loss_coef=0.01
            if args.color_net_type=="naive":
                args.color_net_params={
                        "fin_dim":48,
                        "pin_dim":3,
                        "view_dim":3,
                        "pfin_dim":args.features_dim,
                        "en_dims":[128,96,64],
                        "de_dims":[48,48],
                        "multires":[10,0],
                        "pre_compc":args.use_colors_precomp,
                        "cde_dims":[48],
                        "use_pencoding":[True,False],#postion viewdir
                        "weight_norm":False,
                        "weight_xavier":True,
                        "use_drop_out":True,
                        "use_decode_with_pos":args.use_decode_with_pos,
                    }
            else:
                raise NotImplementedError
    return args
            