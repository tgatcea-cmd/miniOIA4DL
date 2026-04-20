# distutils: extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
# distutils: extra_link_args = ['-fopenmp']
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
## ^^^ DIRECTIVAS generadas por IA ^^^ ##

# -O3: Activa vectorización automática del compilador
# -ffast-math: Permite reordenar operaciones de punto flotante para ganar velocidad
# - boundscheck=False: Elimina la comprobación de límites en cada acceso al array
# - cdivision=True: Usa división de C en lugar de la división de Python que maneja casos de resto de forma distinta

import numpy as np
cimport numpy as np
from cython.parallel cimport prange





### im2col #####################################################################################
'''
Función de transformación "im2col". Desenrrollamos parches de las imagenes para procesarlo por columnas
en paralelo, transformando la operación de convolución en una multiplicación de matrices GEMM.
'''

### Bloque generado por IA ##############################
def im2col_forward_cython(float[:, :, :, ::1] input, int KH, int KW, int stride, int padding):
### Fin bloque generado por IA ##############################

    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]

    # Formula estandar
    cdef int OH = (H + 2 * padding - KH) // stride + 1
    cdef int OW = (W + 2 * padding - KW) // stride + 1

    cdef int row_size = C * KH * KW
    cdef int col_size = B * OH * OW

    # Inicializamos la matriz de salida con ceros para manejar el padding implicitamente
    # y usamos un "memory view" para que Cython pueda tener acceso a aritmetica de punteros
    # gracias al acceso a memoria contigua
    cdef np.ndarray[float, ndim=2] cols = np.zeros((row_size, col_size), dtype=np.float32)
    cdef float[:, ::1] cols_view = cols

    cdef int c, kh, kw, b, oh, ow
    cdef int row, h_in, w_in, oh_offset, b_offset

    with nogil:
        ### Bloque modificado por IA ##############################
        # Paralelizamos los canales de entrada
        for c in prange(C, schedule='static'):

            # Iteramos sobre las dimensiones del kernel
            for kh from 0 <= kh < KH:
                for kw from 0 <= kw < KW:

                    # Calculamos la fila actual en la matriz cols
                    row = (c * KH + kh) * KW + kw

                    for b from 0 <= b < B:

                        # Precalculamos el b_offset para reducir multiplicaciones en el bucle interno
                        b_offset = b * OH * OW

                        for oh from 0 <= oh < OH:

                            # Calculamos la coordenada en la imagen original
                            h_in = oh * stride - padding + kh

                            # Si estamos en los limites de la imagen..
                            if 0 <= h_in < H:

                                # Calculamos la coordenada restante de la imagen original
                                oh_offset = b_offset + oh * OW

                                # Si estamos en los limites de la imagen completa..
                                for ow from 0 <= ow < OW:

                                    # Acceso utilizando stride para acceder de 1 en 1
                                    w_in = ow * stride - padding + kw

                                    # Si estamos en los limites de la imagen..
                                    if 0 <= w_in < W:
                                        cols_view[row, oh_offset + ow] = input[b, c, h_in, w_in]
        ### Fin bloque modificado por IA ##############################
    return cols
########################################################################################
