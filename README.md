# LightMove: A Lightweight Next-POI Recommendation for Taxicab Rooftop Advertising(CIKM21)
 
Mobile digital billboards are an effective way to augment brand awareness. Among various such mobile billboards, taxicab rooftop devices are emerging in the market as a brand new media. Motov is a leading company in South Korea in the taxicab rooftop advertising market. In this work, we present a lightweight yet accurate deep learning-based method to predict taxicabs’ next locations to better prepare for targeted advertising based on demographic information of locations. Considering the fact that next POI recommendation datasets are frequently sparse, we design our presented model based on neural ordinary differential equations (NODEs), which are known to be robust to sparse/incorrect input, with several enhancements. Our model, which we call LightMove, has a larger prediction accuracy, a smaller number of parameters, and/or a smaller training/inference time, when evaluating with various datasets, in comparison with state-of-the-art models.

# Usage
## Install the environment using yaml file
~~~
conda env create --file environment.yaml
~~~
## Model parameter 
- pretrain : 0(Train), 1(Train with pretrained model), 2(Test)
- model_method : 0(G0E) 1(L2E) 2(G2E) 3(G5E)

## Train model 
- Change localpath to your path 
~~~
python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/LightMove/train/ --pretrain 0 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 1 --epoch_max 50 --learning_rate 0.005
~~~

## Train with pretrained model
~~~
python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/LightMove/test/ --pretrain 2 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 1 --epoch_max 50 --learning_rate 0.005
~~~

## Test model
~~~
python main.py --data_name  'foursquare2' --data_path '/localpath/LightMove/data/' --save_path /localpath/LightMove/test/ --pretrain 2 --loc_emb_size 100 --uid_emb_size 60 --tim_emb_size 10 --hidden_size 100 --dropout_p 0.3 --model_method 2 --epoch_max 50 --learning_rate 0.005
~~~

## Authors
Jinsung Jeon, Minju Jo, Seunghyeon Cho, Noseong Park(Yonsei University)<br>
Soyoung Kang(NAVER Clova)<br>
Seonghoon Kim, Chiyoung Song(Motov Inc., Ltd.)
