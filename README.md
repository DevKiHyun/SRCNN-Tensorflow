# SRCNN-Tensorflow (2018/08/15)

## Introduction
We implement a tensorflow model for ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/pdf/1501.00092.pdf)
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
- train.py : Train the SRCNN model and represent the test performance.
- demo.py : Test the SRCNN model and show result images and psnr.
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

### Test
```shell
python test.py

# Default args: image_index = 1, scale = 2, coordinate = [50,50], interval = 30 
# You can change args: image_index = 13, scale = 4, coorindate [100,100], interval = 50

python test.py --image_index 13 --scale 4 --coordinate [100,100] --interval 50
```
### Demo
```shell
python demo.py

# Default args: scale = 2
# You can change arg: scale = 4

python demo.py --scale 4
```

## Result
##### Results on Set 5

|  Scale    | Bicubic | tf_SRCNN |
|:---------:|:-------:|:----:|
| 2x - PSNR|   33.33	|   34.92	|

##### Results on Urban 100 (visual)
- Original (Urban100 / index 1)

  ![Imgur](https://github.com/DevKiHyun/SRCNN-Tensorflow/blob/master/SRCNN/result/original.png)
 
 - Bicubic (Urban100 / index 1)

    ![Imgur](https://github.com/DevKiHyun/SRCNN-Tensorflow/blob/master/SRCNN/result/bicubic.png)
 
 - SRCNN (Urban100 / index 1)
 
    ![Imgur](https://github.com/DevKiHyun/SRCNN-Tensorflow/blob/master/SRCNN/result/SRCNN.png)

