Multiple choice question chatbot

![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/lstm_chatbot.jpg)

data:
![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/dataformat.jpg)

preprocess:
python3.6 make_dataset.py dir/for/data/config.json(dir for data) 

make_dataset.py 
|--embedding.py	//make embedding with word vector and words made by preprocessor.collect_words	 
|--preprocessor.py	//provide preprocess function
  |--dataset.py	//add speaker before sentences can improve model preformace

introdution:
preprocess data to train.pkl,valid.pkl,embedding.pkl for model training

python3.6 make_dataset_test.py test/data/name

introdution:
use ./embedding.pkl to preprocess test data to test.pkl

model:
![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/model_struct.jpg)

train:
python3.6 train.py dir/for/models/config.json

train.py
|--base_predictor.py
| |--example_predictor.py
| |--lstm_predictor.py
| |--atten_predictor.py
|--callbacks.py
|--metrics.py

introduction:
save trained model to model.pkl

predict:
python3.6 predict.py dir/for/models/config.json --epoch x(the save model of which epoch to use)

introduction:
save predict result to predict-x.csv

![image](https://github.com/TaiChunYen/adl_hw1/blob/master/picture/outputformat.jpg)
