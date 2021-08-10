# IF you want to train a model

CUDA_VISIBLE_DEVICES=0 python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/train/ --pretrain 0 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 1 --epoch_max 50 --learning_rate 0.005

# IF you want to test a model
CUDA_VISIBLE_DEVICES=0 python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/test/ --pretrain 1 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 1 --epoch_max 50 --learning_rate 0.005