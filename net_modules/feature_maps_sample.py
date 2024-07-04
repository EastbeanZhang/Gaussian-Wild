import torch


def normalize_pts(pts, box):
    if box.shape[0]==1:
        box=box.squeeze()
        return (pts - box[1,:].unsqueeze(0)) * (2.0 / (box[0,:].unsqueeze(0) - box[1,:].unsqueeze(0))) - 1.0
    elif box.shape[0]>1:
        return (pts - box[:,1,:]) * (2.0 / (box[:,0,:] - box[:,1,:])) - 1.0
    else:
        return None


# def sample_from_feature_maps_v1(feature_maps, pts,box_coord,coord_scale=1,combine_method="cat", mode='bilinear', padding_mode='border'):

#     n_maps, C, H, W = feature_maps.shape #3 _ _ _
#     pts_norm=normalize_pts(pts,box_coord[:,:2,:]).unsqueeze(0)
#     _,M, _ = pts_norm.shape
#     #feature_maps = feature_maps.view(N*n_maps, C, H, W)
#     pts_xy = pts_norm[:,:,[0,1]]
#     pts_xz = pts_norm[:,:, [0, 2]]
#     pts_yz = pts_norm[:,:, [1, 2]]
#     coordinates=torch.cat([pts_xy,pts_xz,pts_yz],dim=0)
#     if n_maps<=3:
#         coordinates=coordinates[:n_maps,:,:]
#     else:
#         pts_new=(box_coord[:,2:n_maps-1,:2]/coord_scale).permute(1,0,2)
#         coordinates=torch.cat([coordinates,pts_new],dim=0)
#     coordinates=coordinates.unsqueeze(1)
#     #M n_maps C
#     output_features = torch.nn.functional.grid_sample(feature_maps, coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(2,3,0,1).squeeze()
#     if combine_method=="cat":
#         output_features=output_features.reshape((output_features.shape[0],-1)) 
#     elif combine_method=="sum":
#         output_features=output_features.sum(dim=1)
#     return output_features,coordinates

def sample_from_feature_maps(feature_maps, pts,box_coord,coord_scale=1,combine_method="cat", mode='bilinear', padding_mode='border'):

    n_maps, C, H, W = feature_maps.shape

    #feature_maps = feature_maps.view(N*n_maps, C, H, W)
    coordinates=box_coord.permute(1,0,2)/coord_scale

    coordinates=coordinates.unsqueeze(1)# n_maps 1 M 2
    #M n_maps C
    output_features = torch.nn.functional.grid_sample(feature_maps, coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(2,3,0,1).squeeze()
    if combine_method=="cat":
        output_features=output_features.reshape((output_features.shape[0],-1)) 
    elif combine_method=="sum":
        output_features=output_features.sum(dim=1)
    return output_features,coordinates

if __name__=="__main__":
    pass
