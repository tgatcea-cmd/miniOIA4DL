# MAIN EXCECUTION ENVIRONMENT
* CPU:  Ryzen 5 3400G
* RAM:  2x8GB RAM 3200Mhz
* GPU:  GTX 1660 6GB
* OS:   ~~Windows 10 pro~~ -> CachyOS
* DISK: ~~nvme m.2 ssd gen3~~ -> HDD

# DAY zero
* FORKED to `tgatcea-cmd` and cloned locally
* CREATED deepwiki of original repo at https://deepwiki.com/adcastel/miniOIA4DL
* EXECUTED `$ python main.py` and got a `ZeroDivisionError: float division by zero` error in the  `.\models\basemodel.py` file, line 22, in function definition `forward`: `images_per_second = imgs / layer_time`
* FIXED `layer_time`:`.\models\basemodel.py`(`forward` function) measurement by using `time.perf_time()` instead of `time.time()`
* EXECUTED `$ python main.py` succesfully: Total time: 73.95s IPS: 0.11images/sec
* MEASURED all available models with default configuration:

    | **Model** | **Excecution Time** | **Performance** | **Configuration** |
    |:---------:|:-------------------:|:---------------:|:-----------------:|
    | OIANet    | 70.17s              | 0.11 images/sec | default           |
    | TinyCNN   | 126.15s             | 0.06 images/sec | default           |
    | AlexNet   | 1103.71s            | 0.01 images/sec | default           |
    | ResNet18  | N/D                 | N/A             | epochs=1          |

* FIXED `resnet18_cifar_100.py` to include `training=False` parameter in function definition `forward(self, x, curr_iter=1)` after getting an unexpected keyword argument 'training' error in `$ python main.py --model ResNet18` execution
* CHANGED time all time measurements in `resnet18_cifar_100.py` to `time.perf_counter()` as prevention for rounding error (recommendation from inline copilot predictor when searching for `forward(self, x, curr_iter=1)` definition)

# DAY one
* EXECUTED and LOGGED `$ python main.py --model OIANet > ./logs/oianet_baseline.log` succesfully.
* ANALYZED 'oianet_baseline.log' and got a clear frontier on comparing layer performance (x62.79):
    * SLOWER LAYERS: Conv2D, Dense, MaxPool2D
    * FASTER LAYERS: BatchNorm2D, ReLU, Dropout, Softmax, Flatten
* EXECUTED and LOGGED `$ python main.py --model TinyCNN > ./logs/tinycnn_baseline.log` succesfully.
* ANALYZED 'tinycnn_baseline.log' and got a clear frontier on comparing layer performance (x9.43):
    * SLOWER LAYERS: Conv2D, Dense
    * FASTER LAYERS: BatchNorm2D, ReLU, Dropout, Softmax, Flatten
* STUDIED `_forward_direct` algorithm on `Conv2D` module.
* ADDED `_forward_im2col_GEMM` algorithm as `conv_algo 1` to `Conv2D` module following the `im2col+GEMM` aproach.
* EXECUTED and LOGGED `$ python main.py --model OIANet --conv_algo 1 > ./logs/oianet_(conv=1).log` succesfully.
* ANALYZED 'oianet_(conv=1).log' and got a clear frontier on comparing layer performance (x5.47):
    * SLOWER LAYERS: Dense, MaxPool2D
    * FASTER LAYERS: Conv2D, BatchNorm2D, ReLU, Dropout, Softmax, Flatten

### FIRST TARGET avg. performance sum per layer >= 1000 imgs/sec with OIANet

# LOST-MEDIA
Due to OS corruption I lost the Days 2 and 3 progress.
## DAY two
Primarely focused in Dense layer and Utils file.
## DAY three
Primarely focused in MaxPool2D layer.

# DAY four
* CREATED script `benchmark_script.py` to average execution results.
* BENCHMARKED `OIANet` as `BASELINE` (all naive) in `./logs/oianet_BASELINE_cachy.csv` with 10 executions.
* ANALYZED `./logs/oianet_BASELINE_cachy.csv` and got a clear frontier on comparing layers performance (x52.59):
![image](./logs/images/layers_oianet_baseline.png)
    * SLOWER LAYERS (avg. perf. sum < 1000 imgs/sec): Conv2D, Dense, MaxPool2D
    * FASTER LAYERS (avg. perf. sum >= 1000 imgs/sec): BatchNorm2D, ReLU, Softmax, Dropout, Flatten
* BENCHMARKED `OIANet` as `conv_algo=1` in `./logs/oianet_(conv=1_others=naive)_cachy.csv` with 10 executions.
* ANALYZED `./logs/oianet_(conv=1_others=naive)_cachy.csv` and got a clear frontier on comparing layers performance (x6.14):
![image](./logs/images/layers_oianet_(conv=1_others=naive).png)
    * SLOWER LAYERS: Dense, MaxPool2D, Conv2D
    * FASTER LAYERS: BatchNorm2D, ReLU, Softmax, Dropout, Flatten
* STUDIED `matmul_biasses` algorithm in `utils.py` used by `Dense` layer.
* ADDED `inline` aproach in the `matmul_biasses` algorithm in `utils.py`.
* BENCHMARKED `OIANet` as `conv_algo=1 dense=inline` in `./logs/oianet_(conv=1_dense=inline_others=naive)_cachy.csv` with 10 executions.
* ANALYZED `./logs/oianet_(conv=1_dense=inline_others=naive)_cachy.csv` and got a clear frontier on comparing layers performance (x6.14):
![image](./logs/images/layers_oianet_(conv=1_dense=inline_others=naive).png)
    * SLOWER LAYERS: MaxPool2D, Conv2D
    * FASTER LAYERS: BatchNorm2D, Dense, ReLU, Softmax, Dropout, Flatten
* STUDIED `forward` algorithm in `MaxPool2D` layer.
* ADDED `vectorized` approach in `forward` algorithm in the `MaxPool2D` layer based on:
    * Nested loops bottleneck: naive looping suffers from data locality. Vectorization maximizes data locality for cache.
    * SIMD: numpy shifts excecution to compiled C-backend, offers performant SIMD micro-kernels
    * Mapping: sliding window approach creates a view from stride manipulation, avoiding main memory duplication and maximizing data locality.
* BENCHMARKED `OIANet` as `conv_algo=1 dense=inline maxpool2d=vectorizado` in `./logs/oianet_(conv=1_dense=inline_maxpool2d=vectorizado_others=naive)_cachy.csv` with 10 executions.
* ANALYZED `./logs/oianet_(conv=1_dense=inline_maxpool2d=vectorizado_others=naive)_cachy.csv` and got a clear frontier on comparing layers performance (x2.42):
![image](./logs/images/layers_oianet_(conv=1_dense=inline_maxpool2d=vectorizado_others=naive).png)
    * SLOWER LAYERS: Conv2D
    * FASTER LAYERS: BatchNorm2D, Dense, ReLU, Softmax, Dropout, Flatten
* DEEP STUDY of `Conv2D` layer.
