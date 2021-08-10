# install the environment using yaml file
# conda env create --file environment.yaml

# Change your localpath 

# if you want to train a model
# CUDA_VISIBLE_DEVICES=0 python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/LightMove/train/ --pretrain 0 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 1 --epoch_max 50 --learning_rate 0.005

# if you want to test a pretrained model
# CUDA_VISIBLE_DEVICES=0 python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/LightMove/test/ --pretrain 1 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 1 --epoch_max 50 --learning_rate 0.005# LightMove
