python tools/predict.py \
       --config configs/drivable_semseg_configs/pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k.yml \
       --model_path train_pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_1000k_2023_X399/best_model_415000/model.pdparams \
       --image_path /home/malx470/phd/deeplab/rgb/00000_rgb_FRONT.png \
       --save_dir output/result
