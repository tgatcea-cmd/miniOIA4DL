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




### MaxPool2D #####################################################################################
'''

'''

### Bloque generado por IA ##############################
def maxpool_forward_cython(float[:, :, :, ::1] input, int KH, int KW, int stride):
### Fin bloque generado por IA ##############################

    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]

    cdef int OH = (H - KH) // stride + 1
    cdef int OW = (W - KW) // stride + 1

    ### Bloque generado por IA ########################
    cdef np.ndarray[float, ndim=4] output = np.empty((B, C, OH, OW), dtype=np.float32)
    cdef np.ndarray[int, ndim=5] max_indices = np.empty((B, C, OH, OW, 2), dtype=np.int32)

    # Creamos vistas de memoria para facilitar acceso de stride 1 en 1
    cdef float[:, :, :, ::1] output_view = output
    cdef int[:, :, :, :, ::1] indices_view = max_indices

    cdef int b, c, oh, ow, kh, kw, h_in, w_in
    cdef float max_val, val
    ### Fin de bloque generado por IA ########################

    with nogil:

        # Paralelizamos ejecución: imagenes entre hilos
        for b in prange(B, schedule='static'):
            for c in range(C):
                for oh in range(OH):
                    for ow in range(OW):

                        max_val = -3.4e38

                        # Para la ventana kernel
                        for kh in range(KH):
                            h_in = oh * stride + kh
                            for kw in range(KW):
                                w_in = ow * stride + kw
                                val = input[b, c, h_in, w_in]

                                ### Guardamos indices si es max #####
                                if val > max_val:
                                    max_val = val

                                    indices_view[b, c, oh, ow, 0] = h_in
                                    indices_view[b, c, oh, ow, 1] = w_in
                                ##########

                        output_view[b, c, oh, ow] = max_val

    return output, max_indices
########################################################################################
