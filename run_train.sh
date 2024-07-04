### If execute the following commands, uncomment them

## sacre_coeur
#CUDA_VISIBLE_DEVICES=2 python ./train.py --source_path /path to/Heritage-Recon/sacre_coeur/dense/ \
 #--scene_name sacre --model_path outputs/sacre/full --eval --resolution 2 --iterations 70000 \

## brandenburg_gate
#CUDA_VISIBLE_DEVICES=2 python ./train.py --source_path /path to/Heritage-Recon/brandenburg_gate/dense/ \
#--scene_name brandenburg --model_path outputs/brandenburg/full --eval --resolution 2 --iterations 70000 \

## trevi_fountain
#CUDA_VISIBLE_DEVICES=2 python ./train.py --source_path /path to/Heritage-Recon/trevi_fountain/dense/ \
#--scene_name trevi --model_path outputs/trevi/full --eval --resolution 2 --iterations 70000 \


###lego dataset

# ## perturbation with color and occlusion
#CUDA_VISIBLE_DEVICES=2 python ./train.py --source_path /path to/NeRF_synthetic/lego/ --scene_name lego --model_path outputs/lego/pertur_color_occ\
# --eval --resolution 1 --iterations 20000  --white_background --data_perturb color occ --use_decode_with_pos  --densify_grad_threshold 0.00015 --features_mask_iters 3000


