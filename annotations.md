# MAIN EXCECUTION ENVIRONMENT
* Ryzen 5 3400G
* 2x8GB RAM 3200Mhz
* GTX 1660 6GB
* Windows 10 pro

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

* FIXED `resnet18_cifar_100.py` to include `training=False` parameter on function definition `forward(self, x, curr_iter=1)` after getting an unexpected keyword argument 'training' error on `$ python main.py --model ResNet18` execution
* CHANGED time all time measurements in `resnet18_cifar_100.py` to `time.perf_counter()` as prevention for rounding error (recommendation from inline copilot predictor when searching for `forward(self, x, curr_iter=1)` definition)

# DAY one
* EXECUTED and LOGGED `$ python main.py --model OIANet > ./logs/oianet_baseline.log` succesfully.
* ANALYZED 'oianet_baseline.log' and got a clear frontier on comparing layer performance (x62.79):
    * SLOWEST LAYERS: Conv2D, Dense, MaxPool2D
    * FASTEST LAYERS: BatchNorm2D, ReLU, Dropout, Softmax, Flatten
* EXECUTED and LOGGED `$ python main.py --model TinyCNN > ./logs/tinycnn_baseline.log` succesfully.
* ANALYZED 'tinycnn_baseline.log' and got a clear frontier on comparing layer performance (x9.43):
    * SLOWEST LAYERS: Conv2D, Dense
    * FASTEST LAYERS: BatchNorm2D, ReLU, Dropout, Softmax, Flatten
