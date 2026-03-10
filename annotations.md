# MAIN EXCECUTION ENVIRONMENT
* Ryzen 5 3400G
* 16GB RAM 3200Mhz
* GTX 1660 6GB
* Windows 10 pro

# DAY zero
* forked to `tgatcea-cmd` and cloned locally
* created deepwiki of original repo at https://deepwiki.com/adcastel/miniOIA4DL
* executed `python main.py` and got no response after a few minutes -> my guess: bad downloading settings
* downloaded from cifar url in "data\PISTA" file
* mannually extracted `.tar.gz` into `cifar-100-python` folder
* executed `python main.py` and got a `ZeroDivisionError: float division by zero` error in the  `.\models\basemodel.py` file, line 22, in function definition `forward`: `images_per_second = imgs / layer_time`
* corrected `layer_time`:`.\models\basemodel.py`(`forward` function) measurement by using `time.perf_time()` instead of `time.time()`
* executed `python main.py` succesfully: Total time: 73.95s IPS: 0.11images/sec
