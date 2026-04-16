# MAIN EXCECUTION ENVIRONMENT

Component | Model 
:--|:--
CPU |  Ryzen 5 3400G
RAM |  2x8GB RAM 3200Mhz
GPU |  GTX 1660 6GB
STORAGE |  ~~nvme m.2 ssd gen3~~ \ HDD
OS  |  ~~Windows 10 pro~~ \ CachyOS

---

# DAILY SUMMARY

## DAY zero
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

## DAY one
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

### FIRST TARGET: **`avg. performance` average per layer >= 1000 imgs/sec** for OIANet

## LOST-MEDIA
Due to OS corruption I lost the Days 2 and 3 progress.
### DAY two
Primarely focused in Dense layer and Utils file.
### DAY three
Primarely focused in MaxPool2D layer.

## DAY four
Worked on retrieving as much as possible from previous progress.
* CREATED script `benchmark_script.py` to average execution results.
* BENCHMARKED `OIANet` as `BASELINE` (all naive) in `./logs/oianet_BASELINE_cachy.csv` with 10 executions.
* ANALYZED `./logs/oianet_BASELINE_cachy.csv` and got a clear frontier on comparing layers performance (x52.59):

    ![image](./logs/images/layers_oianet_baseline.png)
    * SLOWER LAYERS (avg. perf. average < 1000 imgs/sec): Conv2D, Dense, MaxPool2D
    * FASTER LAYERS (avg. perf. average >= 1000 imgs/sec): BatchNorm2D, ReLU, Softmax, Dropout, Flatten
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
* DEEP STUDY of `Conv2D` layer

---
# ARCHITECTURAL OVERHAUL SUMMARY
Performance optimizations applied sequentially across framework layers to eliminate Python bottlenecks and maximize hardware Arithmetic Intensity. 

Performance Trajectory (OIANet):
* Day 0 (Baseline): 0.11 images/sec. Interpreter bounded.
* Day 4 (Optimized): 120.85 images/sec. Layers strictly routed to C-compiled BLAS and SIMD micro-kernels. `Conv2D`, `Dense`, and `MaxPool2D` upgraded from naive loops to vectorized execution.


## Module-Level Documentation

### 1. Layer: `Conv2D` (conv2d.py)

Handles spatial convolutions. Implements three algorithmic tiers controllable via the `conv_algo` parameter.

* Mode 0: Direct (Naive)
    * Mechanism: 7-level nested loop iterating over $[Batch, Out\_C, In\_C, Out\_H, Out\_W, K_H, K_W]$.
    * Status: Deprecated. Arithmetic intensity destroyed by scalar operations and dynamic type-checking in Python.
* Mode 1: IM2COL + GEMM (Python Loops)
    * Mechanism: Flattens spatial patches into a matrix using 3-level loops and `cols.append()`. Computes via `@` operator.
    * Status: Suboptimal. Matrix multiplication is fast (BLAS), but memory fragmentation and Python list reallocation during `im2col` data preparation bottleneck execution.
* Mode 2: IM2COL + GEMM + Vectorized
    * Mechanism: Zero Python loops. 
        1.  `np.lib.stride_tricks.as_strided` generates a 6D virtual tensor view $[B, C, OH, OW, K_H, K_W]$ using pointer arithmetic. Zero initial memory duplication.
        2.  `.transpose().reshape(-1)` flattens the view into standard IM2COL matrices.
        3.  `kernel @ cols` delegates computation to BLAS.
    * Status: Optimal. Maximizes L1/L2 cache utilization.

### 2. Layer: `MaxPool2D` (maxpool2d.py)

Executes spatial downsampling and caches gradient routing indices.

* Naive Approach: 4-level loop $[B, C, OH, OW]$ extracting windows and computing scalar maximums. Interpreter overhead saturates CPU.
* Vectorized Approach:
    * Forward Pass: Utilizes `sliding_window_view` to map overlapping patches. A single `np.max(windows, axis=(4, 5))` command shifts execution to SIMD reductions.
    * Training State: `np.argmax` extracts 1D indices from a flattened view. Coordinates are translated to absolute grid coordinates via tensor broadcasting (`abs_i = max_i + grid_i`). Allows strictly parallelized tracking of active gradients.

### 3. Layer: `Dense` (dense.py, utils.py)

Linear transformation mapping $Y = XW^T + b$.

* Naive Approach: 3-level nested loop $O(M \cdot N \cdot P)$. Highly susceptible to memory latency.
* Optimized Inline Approach: 
    * Replaced explicit loops with `np.matmul(A, B, out=C)`. 
    * Memory Efficiency: The `out=C` parameter forces in-place modification, preventing the allocation of intermediate ghost arrays. Bias is added via vectorized broadcasting.

---

### Implementation Directives
* All active mathematical transformations utilize `np.float32` for memory alignment and numerical stability across BLAS parameters.
* Execution strictly adheres to NCHW dimensional format at inputs and outputs, transposing intermediate IM2COL/GEMM geometries only during computation.