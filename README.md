# SRCNN-Tensorflow (2018/08/15)

## Introduction
We implement a tensorflow model for ["Image Super-Resolution Using Deep Convolutional Networks"]
(https://arxiv.org/pdf/1501.00092.pdf)
 - We use 91 dataset as training dataset.
 
## Environment
- Ubuntu 16.04
- Python 3.5

## Depenency
- Numpy
- Opencv2
- matplotlib

## Files
- main.py : Execute train.py and pass the default value.
- srcnn.py : srcnn model definition.
- train.py : Train the srcnn model and represent the test performance.
- util.py : Utility functions for this project.
- log.txt : The log of training process.
- model : The save files of the trained srcnn.

## How to use
### Training
```shell
python main.py

# if you want to change training epoch ex) 1500 epoch (default) -> 2000 epoch
python main.py --training_epoch 2000
```
