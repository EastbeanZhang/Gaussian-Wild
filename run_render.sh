
CUDA_VISIBLE_DEVICES=0 python ./render.py  --model_path outputs/sacre/full
CUDA_VISIBLE_DEVICES=0 python ./render.py  --model_path outputs/sacre/full --skip_train --skip_test --render_multiview_vedio
CUDA_VISIBLE_DEVICES=0 python ./render.py  --model_path outputs/sacre/full --skip_train --skip_test --render_interpolate