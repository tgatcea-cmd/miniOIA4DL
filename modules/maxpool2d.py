from modules.layer import Layer
import numpy as np

### Funciones implementadas en Cython ########################
from cython_modules.maxpool2d import maxpool_forward_cython
######################################

### Librería para Vectorización ##########################
from numpy.lib.stride_tricks import sliding_window_view
#####################################


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride, maxpool2d_algo=0):
        self.kernel_size = kernel_size
        self.stride = stride

        if maxpool2d_algo == 0:
            self.mode = 'naive' 
        elif maxpool2d_algo == 1:
            self.mode = 'vectorization'
        elif maxpool2d_algo == 2:
            self.mode = 'cython'
        else:
            print(f"Algoritmo {maxpool2d_algo} no soportado aún. Usando vectorización.")
            self.mode = 'vectorization'

    def forward(self, input, training=True):
        if self.mode == 'naive':
            return self._forward_naive(input)
        elif self.mode == 'vectorization':
            return self._forward_vectorization(input, training)
        elif self.mode == 'cython':
            return self._forward_cython(input)
        else:
            raise ValueError("Error maxpool2d mode")


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





    ### Naive ##########################################    
    def _forward_naive(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        # Cálculo de dimensiones de salida según la fórmula estándar de pooling
        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # Almacenamos los índices del máximo para el backpropagation
        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w), dtype=input.dtype)

        # Iteración sobre todas las dimensiones
        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        
                        # Definimos los límites de la ventana actual
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        # Extraemos la ventana y buscamos el valor máximo y su posición
                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        # Guardamos el resultado y el índice global
                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])
        
        return output
    #############################################






    ### Vectorization ##########################################
    # Optimizamos usando los "stride tricks" de numpy para evitar bucles explícitos
    def _forward_vectorization(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # Creamos una vista sliding_window_view basada en ventanas
        # sin duplicar datos en memoria, manipularemos los strides
        # axis(2, 3) -> [H, W]
        windows = sliding_window_view(input, window_shape=(KH, KW), axis=(2, 3))
        # Aplicamos el stride saltando elementos en la vista generada
        windows = windows[:, :, ::SH, ::SW, :, :]

        # Calculamos el máximo sobre los ejes de la ventana (4 y 5)
        output = np.max(windows, axis=(4, 5))

        ### Bloque generado por IA ##################3
        if training:
            # Aplanamos las ventanas para usar argmax unidimensional
            windows_flat = windows.reshape(B, C, out_h, out_w, -1)
            max_idx_flat = np.argmax(windows_flat, axis=-1)

            # Convertimos el índice plano a coordenadas 2D locales al parche
            max_i, max_j = np.unravel_index(max_idx_flat, (KH, KW))

            # Creamos rejillas de anclaje para convertir coordenadas locales a globales
            grid_i = np.arange(out_h) * SH
            grid_j = np.arange(out_w) * SW

            # Broadcasting para sumar los offsets globales a los índices locales
            abs_i = max_i + grid_i.reshape(1, 1, out_h, 1)
            abs_j = max_j + grid_j.reshape(1, 1, 1, out_w)

            # Empaquetamos los índices absolutos en un tensor [B, C, H_out, W_out, 2]
            self.max_indices = np.stack((abs_i, abs_j), axis=-1)
        ### Fin bloque generado por IA ##################

        return output
    #############################################





    ### Cython ##########################################
    # Eliminamos el overhead del interpreter y habilitamos paralelismo multi-hilo
    def _forward_cython(self, input):
        self.input = input
        
        # Aseguramos que la memoria sea contigua para que C pueda acceder vía punteros lineales
        input_cython = np.ascontiguousarray(input, dtype=np.float32)
        
        # Ejecutamos la función externa en Cython
        output, self.max_indices = maxpool_forward_cython(input_cython, self.kernel_size, self.kernel_size, self.stride)
        
        return output
    #############################################

