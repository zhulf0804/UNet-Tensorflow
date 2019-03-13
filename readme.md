## About

+ A brief ***UNet tensorflow*** implementation. It can work well on our dataset, see images below. **If data augmentation and more strategies,
the performance will be better**.
![test.png](./test.png)
+ You just need to config the **config.py** to fit your own datast, see [dataset](#dataset). When the configuration is finished, you just to run and test.
+ The code will be updated with namescope, tfrecord, and more summaries.

    

## Environment

+ Anaconda(python 2.7)
+ Tensorflow 1.10

## <span id = 'dataset'>Dataset</span>

The dataset can be organized as follows:

```
|-- data_path
        |-- img_dir_name
        |-- annotation_dir_name
        |-- train_list_file
        |-- trainval_list_file

```

## Train

> python train.py

## Test

> python test.py


