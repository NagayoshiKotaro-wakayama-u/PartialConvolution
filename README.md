# PartialConvolution

## generateToyData.py
To generate ToyData run this command.   
`python generateToyData.py`  
Then 1500 training images and 100 validation images, 100 test images (and each mask images) are generated.   
  
↓ToyData example↓  
<img src="./data/sample.png" width="256px">  
  
Number of images are able to be changed by `-train` and `-valid`, `-test` option.  
For example `python generateToyData.py -train 5000 -valid 500 -test 500`  
  
You can get more details, use `-h` option.  
`python generateToyData.py -h`  

## main.py  
To start training run this command. (xxxx is name of experiment)  
`python main.py xxxx`  
For example `python main.py hogehoge`.  
  
The following directory `experiment` is automatically created when you run this command.  
And the overall directory structure is as follows. (dataSet must be prepared by generateToyData.py or yourself)
```
PartialConvolution/
                 ┣ experiment/
                 ┃    ┗ hogehoge_logs
                 ┃        ┣ logs/
                 ┃        ┣ losses/
                 ┃        ┗ test_samples/
                 ┣ data/
                 ┃    ┗ dataSet/
                 ┃        ┣ train/
                 ┃        ┃   ┗ train_img/
                 ┃        ┣ train_mask/
                 ┃        ┣ test/
                 ┃        ┃   ┗ test_img/
                 ┃        ┣ test_mask/
                 ┃        ┣ valid/
                 ┃        ┃   ┗ valid_img/
                 ┃        ┗ valid_mask/
                 ┣ libs/
                 ┃    ┣ pconv_layer.py
                 ┃    ┣ pconv_model.py
                 ┃    ┗ util.py
                 ┣ main.py
                 ┣ test.py
                 ┗ generateToyData.py
```
  
If you want to try normal PartialConvolution, use `-KLoff` option.  
`python main.py xxxx -KLoff`  
  
You can get more details, use `-h` option.   
`python main.py -h`  
  
  
## test.py  
After training, run this command to test the model.  
`python test.py xxxx yyyy`  
xxxx is name of experiment. yyyy is name of the weight file to be loaded.  
For example
`python test.py KLPConv weights.100-0.13.h5`
  
The directory `result` is automatically created under `experiment` when you run this command.  
You can get more details, use `-h` option.  
  
