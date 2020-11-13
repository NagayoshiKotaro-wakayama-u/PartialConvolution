# PartialConvolution

## main.py  
To start training run this command. (xxxx is name of experiment)  
`python main.py xxxx`  
For example `python main.py KLPConv`.  
  
The following directory `experiment` is automatically created when you run this command.
And the overall directory structure is as follows.
```
PartialConvolution/
                 ┣ experiment/
                 ┃    ┗ KLPConv_logs
                 ┃        ┣ logs/
                 ┃        ┣ losses/
                 ┃        ┗ test_samples/
                 ┣ data/
                 ┃    ┗ dataSet/
                 ┣ libs/
                 ┃    ┣ pconv_layer.py
                 ┃    ┣ pconv_model.py
                 ┃    ┗ util.py
                 ┣ main.py
                 ┗ test.py
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
  
