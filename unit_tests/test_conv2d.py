### Este archivo ha sido modificado por IA para añadir las opciones de algoritmos correspondientes ###

import numpy as np
from modules.conv2d import Conv2D


def test_conv2d():
    # Parameters
    img_width = 5
    img_height = 5
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2

    # Input: 2 images, 1 channel, 5x5 values from 0 to 49
    input_image = np.arange(img_height*img_height*in_channels*batch_size, dtype=np.float32).reshape(batch_size, in_channels, img_width, img_height)

    # Compute expected output manually once
    padded = np.pad(input_image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    out_h = (padded.shape[2] - kernel_size) // stride + 1
    out_w = (padded.shape[3] - kernel_size) // stride + 1
    expected_output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    patch = padded[b, 0, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    expected_output[b, c, i, j] = np.sum(patch)  # kernel is all ones

    # Test each algorithm
    algos = {0: "Direct", 1: "im2col_GEMM", 2: "im2col_GEMM_vectorization", 3: "im2col_GEMM_cython"}
    
    for algo_code, algo_name in algos.items():
        conv = Conv2D(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, conv_algo=algo_code)
        
        # Set all kernels to 1 and biases to 0
        conv.kernels = np.ones((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        conv.biases = np.zeros(out_channels, dtype=np.float32)

        output = conv.forward(input_image)
        assert np.allclose(output, expected_output, atol=1e-5), f"Conv2D {algo_name} mismatch!"
        print(f"✅ Conv2D {algo_name} test passed!")

test_conv2d()


