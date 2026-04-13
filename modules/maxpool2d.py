from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        ### NAIVE ####################################
        # self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        # output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        # for b in range(B):
        #     for c in range(C):
        #         for i in range(out_h):
        #             for j in range(out_w):
        #                 h_start = i * SH
        #                 h_end = h_start + KH
        #                 w_start = j * SW
        #                 w_end = w_start + KW

        #                 window = input[b, c, h_start:h_end, w_start:w_end]
        #                 max_idx = np.unravel_index(np.argmax(window), window.shape)
        #                 max_val = window[max_idx]

        #                 output[b, c, i, j] = max_val
        #                 self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])
        #######################################
    

  
        ### VECTORIZADO ####################################
        # Manipulación directa de punteros, creamos parches "superpuestos" sin duplicar datos.
        # axis(0=Batch, 1=Canal, 2=out_h, 3=out_w, 4=KH, 5=KW)
        windows = sliding_window_view(input, window_shape=(KH, KW), axis=(2, 3))
        # El particionado "::SH, ::SW" aplica el stride.
        windows = windows[:, :, ::SH, ::SW, :, :]

        output = np.max(windows, axis=(4, 5))

        ### Bloque Generado por IA ###
        if training:
            # 2D -> 1D
            windows_flat = windows.reshape(B, C, out_h, out_w, -1)
            # Encuentra la posición unidimensional del máximo.
            max_idx_flat = np.argmax(windows_flat, axis=-1)

            # Obtenemos las coordenadas 2D relativas al parche a partir del indice unidimensional.
            max_i, max_j = np.unravel_index(max_idx_flat, (KH, KW))

            # Genera vectores de anclaje. Representan la coordenada global de cada parche en la imagen original.
            grid_i = np.arange(out_h) * SH
            grid_j = np.arange(out_w) * SW
        ### Fin de bloque generado por IA ###

            # Sumamos los offsets globales a las posiciones relativas locales.
            # Queremos expandir las sumas a lo largo de los Batch y Canales en paralelo
            abs_i = max_i + grid_i.reshape(1, 1, out_h, 1)
            abs_j = max_j + grid_j.reshape(1, 1, 1, out_w)

            # Obtenemos una matriz de Indices Absolutos.
            self.max_indices = np.stack((abs_i, abs_j), axis=-1)
        #######################################

        return output


    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input
