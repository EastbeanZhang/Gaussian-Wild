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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import numpy as np

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,\
    other_viewpoint_camera=None,store_cache=False,use_cache=False,point_features=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if other_viewpoint_camera is not None:#render using other camera center
        viewpoint_camera.camera_center=other_viewpoint_camera.camera_center
    if use_cache:
        pc.forward_cache(viewpoint_camera)
    elif point_features is not None:
        pc.forward_interpolate(viewpoint_camera,point_features)
    else:
        pc.forward(viewpoint_camera,store_cache)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0     #[Npoint,3]
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    if other_viewpoint_camera is not None:#render using other camera center
        viewpoint_camera=other_viewpoint_camera
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    camera_center=viewpoint_camera.camera_center

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,  #4*4
        projmatrix=viewpoint_camera.full_proj_transform,   #4*4
        sh_degree=pc.active_sh_degree,                    #0,
        campos=camera_center,            #3
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz_dealed            #
    means2D = screenspace_points    #[Npoint,3]
    opacity = pc.get_opacity_dealed       #[Npoint,1]  

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling        #[Npoint,3] 
        rotations = pc.get_rotation   #[Npoint,4]   

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if not pipe.convert_SHs_python and pc.use_colors_precomp:
        override_color=pc.get_colors
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features_dealed.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)     #N,3,16  
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))   #N,3  
            #dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            dir_pp_normalized=pc.view_direction
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)      
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)          
        else:
            shs = pc.get_features_dealed
    else:
        colors_precomp = override_color
        

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,       # [Npoint,3]
        means2D = means2D,      #[Npoint,3]
        shs = shs,              #[Npoint,16,3]
        colors_precomp = colors_precomp,
        opacities = opacity,     #[Npoint,1]  
        scales = scales,            #[Npoint,3] 
        rotations = rotations,      #[Npoint,4]    
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}        #N

