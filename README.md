# Multiple choice question chatbot
========

![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/lstm_chatbot.jpg)

## data:
_______
![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/dataformat.jpg)

## preprocess:
____
python3.6 make_dataset.py dir/for/data/config.json(dir for data) 

make_dataset.py 
<p align="left">|--embedding.py(make embedding with word vector and words made by preprocessor.collect_words)</p>
<p align="left">|--preprocessor.py(provide preprocess function)</p>
<p align="left">| |--dataset.py(add speaker before sentences can improve model preformace)</p>

introdution:
preprocess data to train.pkl,valid.pkl,embedding.pkl for model training

python3.6 make_dataset_test.py test/data/name

introdution:
use ./embedding.pkl to preprocess test data to test.pkl

## model:
____
![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/model_struct.jpg)

## train:
___
python3.6 train.py dir/for/models/config.json

train.py
<p align="left">|--base_predictor.py</p>
<p align="left">| |--example_predictor.py</p>
<p align="left">| |--lstm_predictor.py</p>
<p align="left">| |--atten_predictor.py</p>
<p align="left">|--callbacks.py</p>
<p align="left">|--metrics.py</p>

introduction:
save trained model to model.pkl

## predict:
__________________________
python3.6 predict.py dir/for/models/config.json --epoch x(the save model of which epoch to use)

introduction:
save predict result to predict-x.csv

![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/outputformat.jpg)
