CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train_amp.py \
--img_root /nfs_dongmin/VisualGenome/VG_100K \
--model_path model_params/visual_text_79.pth.tar \
--batch_size 4 \
--amp \
--caption_step