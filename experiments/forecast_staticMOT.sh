cd src
python train.py final0110_nopast --exp_id forecasting_staticmot --data_cfg '../src/lib/cfg/mot_seq_static_train.json' --lstm_arch 'gru_decoder_0110_nopast' --encoder_arch 'gru_encoder_0110_nopast' --batch_size 4 --warmup_len 3 --future_len 3 --step_size 10 --seq_len 5 --num_epochs 30 --load_model '../models/ctdet_coco_dla_2x.pth' --save_all
cd ..

#for validation
#python track.py final0110_nopast --load_model ~/mymodel.pth --conf_thres 0.4 --decoder_arch 'gru_decoder_0110_nopast' --encoder_arch 'gru_encoder_0110_nopast' --warmup_len 3 --future_len 3 --step_size 10 --static_val True --exp_name 'myname' --lstm --run_model --save_flow --vis_limit --poly