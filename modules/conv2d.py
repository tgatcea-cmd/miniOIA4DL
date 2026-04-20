from modules.layer import Layer
from modules.utils import *

### Funciones implementadas en Cython ####################
from cython_modules.im2col import im2col_forward_cython
from cython_modules.gemm import gemm_forward_cython
################################

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # MODIFICAR: Añadir nuevo if-else para otros algoritmos de convolución
        if conv_algo == 0:
            self.mode = 'direct' 
        elif conv_algo == 1:
            self.mode = 'im2col_gemm'
        elif conv_algo == 2:
            self.mode = 'im2col_GEMM_vectorization'
        elif conv_algo == 3:
            self.mode = 'im2col_GEMM_cython'
        else:
            print(f"Algoritmo {conv_algo} no soportado aún")
            self.mode = 'direct'

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        # self.mc = 480
        # self.nc = 3072
        # self.kc = 384
        # self.mr = 32
        # self.nr = 12
        # self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        # self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
    
    def forward(self, input, training=True):
        self.input = input
        # PISTA: Usar estos if-else si implementas más algoritmos de convolución
        if self.mode == 'direct':
            return self._forward_direct(input)
        if self.mode == 'im2col_gemm':
            return self._forward_im2col_GEMM(input)
        if self.mode == 'im2col_GEMM_vectorization':
            return self._forward_im2col_GEMM_vectorization(input)
        if self.mode == 'im2col_GEMM_cython':
            return self._forward_im2col_GEMM_cython(input)
        else:
            raise ValueError("Mode must be 'direct' | 'im2col_gemm' | 'im2col_GEMM_vectorization' | 'im2col_GEMM_cython'")

    def backward(self, grad_output, learning_rate):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---

    """
    def conv2d(x, w):
        H, W = x.shape
        K = w.shape[0]

        H_out = H - K + 1
        W_out = W - K + 1

        y = np.zeros((H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                for ki in range(K):
                    for kj in range(K):
                        y[i, j] += x[i + ki, j + kj] * w[ki, kj]
        
        return y
    """

    def _forward_direct(self, input):

        # FORMATO NCHW: Numero de Imagenes, Canales, Altura, Anchura
        batch_size, _, in_h, in_w = input.shape
        # Estas serán las dimensiones del kernel, cuadrado en este caso
        k_h, k_w = self.kernel_size, self.kernel_size


        # añadir padding si es necesario. No queremos que la imagen se encoja demasiado y perder información
        if self.padding > 0:
            # np.pad indica cuanto padding añadir a cada dimensión
            input = np.pad(input,
                           # (0,0) significa que no rellenamos ni el batch ni los canales
                           # (self.padding, self.padding) significa que añadimos padding a la altura y anchura
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)

        # calcular las dimensiones de salida: cuantas veces puede entrar el kernel en la imagen de entrada con el stride dado
        # Matematicamente O = floor[(W - K + 2P) / S] + 1
        # pero el padding ya está sumado implicitamente
        # "//" es la división entera, que equivale a floor para números positivos
        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)


        # Aqui es donde debemos OPTIMIZAR
        # para cada imagen en el batch individualmente
        for b in range(batch_size):
            # para cada canal de salida
            for out_c in range(self.out_channels):
                # para cada canal de entrada, normalmente 3 para RGB
                for in_c in range(self.in_channels):
                    # para cada posición del kernel en la imagen de entrada
                    for i in range(out_h):
                        # para cada posición del kernel en la imagen de entrada
                        for j in range(out_w):
                            # extraer la región de la imagen de entrada que corresponde a la posición actual del kernel
                            # la region es un bloque de mismo tamaño que el kernel, que se va multiplicando
                            # elemento a elemento con el kernel. Luego se suman todos los valores para
                            # obtener el valor de salida para esa posición
                            region = input[b, in_c,
                                           i * self.stride:i * self.stride + k_h,
                                           j * self.stride:j * self.stride + k_w]
                            # como iteramos por los canales de entrada, sumamos los resultados
                            # para formar el pixel final de la salida
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                
                # añadir el sesgo al valor de salida
                output[b, out_c] += self.biases[out_c]

        return output

    def _backward_direct(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

    # PISTA: Se te ocurren otros algoritmos de convolución?





    ### im2col + GEMM ##########################################
    def im2col(self, input):
        batch_size, _, H, W = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride, padding = self.stride, self.padding

        # Extendemos los bordes del tensor para mantener la dimensión espacial
        # sin padding, cada convolución reduce H y W en (kernel-1) píxeles
        if padding > 0:
            input = np.pad(input, ((0,0),(0,0),(padding,padding),(padding,padding)),
                        mode='constant').astype(np.float32)

        # Calculamos las dimensiones de salida: fórmula estándar O = floor((H - K + 2P) / S) + 1
        H_out = (H + 2*padding - k_h) // stride + 1
        W_out = (W + 2*padding - k_w) // stride + 1

        cols = []

        # Extraemos los parches:
        #    -> por cada posición (i,j) del mapa de salida
        #    -> recortamos un bloque [C, K_h, K_w] del input original
        #    -> lo aplanamos
        for b in range(batch_size):
            for i in range(H_out):
                for j in range(W_out):
                    patch = input[b, :, i*stride : i*stride+k_h, j*stride : j*stride+k_w]
                    cols.append(patch.reshape(-1))  # aplanar [C,K,K] → [C·K·K]

        # Transponemos cada columna para que sean respectivamente un parche aplanado
        # -> [C*K*K, B*H_out*W_out]
        return np.array(cols).T


    def _forward_im2col_GEMM(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride, padding = self.stride, self.padding

        # Convertimos la entrada en columnas de parches
        cols = self.im2col(input)

        # Aplanamos los filtos de [Out_C, In_C, K, K] a [Out_C, C*K*K].
        # Cada fila será finalmente un filtro completo "desenrollado"
        kernel = self.kernels.reshape(self.out_channels, -1)

        # Una sola multiplicación matricial reemplaza todas las convoluciones
        # [Out_C, C*K*K] @ [C*K*K, B*Ho*Wo] → [Out_C, B*Ho*Wo]
        out_cols = kernel @ cols
        out_cols += self.biases.reshape(-1, 1)  # bias: broadcasting sobre columnas

        # Devolvemos el formato 4D estándar NCHW: [B, Out_C, H_out, W_out]
        out_h = (in_h + 2*padding - k_h) // stride + 1
        out_w = (in_w + 2*padding - k_w) // stride + 1
        output = out_cols.reshape(self.out_channels, batch_size, out_h, out_w)
        return output.transpose(1, 0, 2, 3)
    ### Fin de im2col + GEMM ##########################################





    ### im2col + GEMM + vectorización ##########################################
    def im2col_vectorization(self, input):
        batch_size, c, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride, padding = self.stride, self.padding

        if padding > 0:
            input = np.pad(input, ((0,0),(0,0),(padding,padding),(padding,padding)), 
                        mode='constant').astype(np.float32)

        out_h = (in_h + 2*padding - k_h) // stride + 1
        out_w = (in_w + 2*padding - k_w) // stride + 1

        # Definir un tensor de 6 dimensiones sin copiar datos: [B, C, H_out, W_out, K_h, K_w]
        # las 4 primeras indican "qué parche"
        # las 2 últimas "posición dentro del parche"
        shape = (batch_size, c, out_h, out_w, k_h, k_w)

        ### Bloque generado por IA #############
        # Le dice a NumPy cuántos bytes saltar en cada eje.
        # No se copia memoria, solo se reinterpreta el buffer existente.
        strides = (
            input.strides[0],            # salto entre batches
            input.strides[1],            # salto entre canales
            input.strides[2] * stride,   # salto vertical entre parches
            input.strides[3] * stride,   # salto horizontal entre parches
            input.strides[2],            # salto vertical DENTRO del parche
            input.strides[3]             # salto horizontal DENTRO del parche
        )

        # as_strided crea la vista 6D
        windows = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)    

        # Permutamos los ejes para que las dimensiones a colapsar queden juntas
        # [C, K_h, K_w, B, H_out, W_out]
        windows_transposed = windows.transpose(1, 4, 5, 0, 2, 3)
        ### Fin de bloque generado por IA #############

        # Aplanar en matriz [C*K*K, B*H_out*W_out]
        return windows_transposed.reshape(c * k_h * k_w, -1)


    def _forward_im2col_GEMM_vectorization(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride, padding = self.stride, self.padding

        # Convertimos la entrada en columnas de parches (vVectorizada)
        cols = self.im2col_vectorization(input)

        # Aplanamos los filtos de [Out_C, In_C, K, K] a [Out_C, C*K*K].
        # Cada fila será finalmente un filtro completo "desenrollado"
        kernel = self.kernels.reshape(self.out_channels, -1)

        # Una sola multiplicación matricial reemplaza todas las convoluciones
        # [Out_C, C*K*K] @ [C*K*K, B*Ho*Wo] → [Out_C, B*Ho*Wo]
        out_cols = kernel @ cols
        out_cols += self.biases.reshape(-1, 1)  # bias: broadcasting sobre columnas

        # Devolvemos el formato 4D estándar NCHW: [B, Out_C, H_out, W_out]
        out_h = (in_h + 2*padding - k_h) // stride + 1
        out_w = (in_w + 2*padding - k_w) // stride + 1
        output = out_cols.reshape(self.out_channels, batch_size, out_h, out_w)
        return output.transpose(1, 0, 2, 3)
    ### Fin de im2col + GEMM + vectorización ##########################################





    ### (im2col + GEMM) @ cython ##########################################
    # ANOTACION: En esta versión es preferible declarar el tipo np.float32 para evitar
    # problemas de interpretación en la parte C.
    def _forward_im2col_GEMM_cython(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride, padding = self.stride, self.padding

        # Forzamos una organización contigua en memoria antes de pasar a C,
        # pues Cython lo requiere para acceso directo al puntero
        input_cython = np.ascontiguousarray(input, dtype=np.float32)

        # Convertimos la entrada en columnas de parches (vCython)
        cols = im2col_forward_cython(input_cython, k_h, k_w, stride, padding).astype(np.float32)

        # Aplanamos los filtos de [Out_C, In_C, K, K] a [Out_C, C*K*K].
        # Cada fila será finalmente un filtro completo "desenrollado".
        kernel = self.kernels.reshape(self.out_channels, -1).astype(np.float32)

        # GEMM también en Cython: evita overhead de numpy para matrices float32 pequeñas
        out_cols = np.zeros((kernel.shape[0], cols.shape[1]), dtype=np.float32)
        gemm_forward_cython(kernel, cols, out_cols) # escribe el resultado en out_cols

        out_cols += self.biases.reshape(-1, 1) # bias: broadcasting sobre columnas

        # Devolvemos el formato 4D estándar NCHW: [B, Out_C, H_out, W_out]
        out_h = (in_h + 2*padding - k_h) // stride + 1
        out_w = (in_w + 2*padding - k_w) // stride + 1
        output = out_cols.reshape(self.out_channels, batch_size, out_h, out_w)
        return output.transpose(1, 0, 2, 3)
    ### Fin de (im2col + GEMM) @ cython ##########################################
        
