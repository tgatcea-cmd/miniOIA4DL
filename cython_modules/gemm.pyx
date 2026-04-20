# distutils: extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
# distutils: extra_link_args = ['-fopenmp']
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
## ^^^ DIRECTIVAS generadas por IA ^^^ ##

# -O3: Activa vectorización automática del compilador
# -ffast-math: Permite reordenar operaciones de punto flotante para ganar velocidad
# - boundscheck=False: Elimina la comprobación de límites en cada acceso al array
# - cdivision=True: Usa división de C en lugar de la división de Python que maneja casos de resto de forma distinta

from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
import numpy as np

cdef enum:
    MR = 6
    NR = 16




cdef inline int imin(int a, int b) noexcept nogil:
    return a if a < b else b




### Empaquetado ###################################
'''
Copian bloques de las matrices A y B a buffers temporales,
reorganizando en micro paneles consecutivos a nivel micro-kernel.
Crucial para las cargas SIMD.

ANOTACION:
pack_A y pack_B reciben ahora 'mr_real' y 'nr_real' — el tamaño
REAL del bloque de borde, que puede ser < mr/nr. El buffer Ac/Bc se rellena
con ceros en las posiciones fantasma para que el micro_kernel nunca lea
basura ni escriba fuera de C.
'''

### Bloque generado por IA ############## <- Errata gorda: IA creia que era Column-major.
cdef void pack_A(float* A, float* Ac, int lda, int mc, int kc, int mr, int mr_real) noexcept nogil:
    cdef int i, k, m
    cdef int idx = 0
    for i from 0 <= i < mc by mr:
        for k from 0 <= k < kc:
            for m from 0 <= m < mr:
                # Relleno con 0 si la fila real no existe (bloque de borde en M)
                if m < mr_real:
                    Ac[idx] = A[(i + m) * lda + k]
                else:
                    Ac[idx] = 0.0
                idx += 1

cdef void pack_B(float* B, float* Bc, int ldb, int kc, int nc, int nr, int nr_real) noexcept nogil:
    cdef int j, k, n
    cdef int idx = 0
    for j from 0 <= j < nc by nr:
        for k from 0 <= k < kc:
            for n from 0 <= n < nr:
                # Relleno con 0 si la columna real no existe (bloque de borde en N)
                if n < nr_real:
                    Bc[idx] = B[k * ldb + (j + n)]
                else:
                    Bc[idx] = 0.0
                idx += 1
### Fin de bloque generado por IA ########
######################################


### Micro-kernel #####################
'''
Recibe los punteros de los empaquetados A y B y calcula
un bloque tamaño MR x NR de la matriz C.
'''

cdef void micro_kernel(float* Ac, float* Bc, float* C, int ldc, int kc) noexcept nogil:
    cdef float ab[MR][NR]
    cdef int i, j, k

    for i from 0 <= i < MR:
        for j from 0 <= j < NR:
            ab[i][j] = 0.0

    for k from 0 <= k < kc:
        for i from 0 <= i < MR:
            for j from 0 <= j < NR:
                ab[i][j] += Ac[k * MR + i] * Bc[k * NR + j]

    for i from 0 <= i < MR:
        for j from 0 <= j < NR:
            C[i * ldc + j] += ab[i][j]

### Bloque generado por IA ############
cdef void micro_kernel_edge(float* Ac, float* Bc, float* C, int ldc, int kc, int mr_real, int nr_real) noexcept nogil:
    cdef float ab[MR][NR]
    cdef int i, j, k

    for i from 0 <= i < MR:
        for j from 0 <= j < NR:
            ab[i][j] = 0.0

    for k from 0 <= k < kc:
        for i from 0 <= i < MR:
            for j from 0 <= j < NR:
                ab[i][j] += Ac[k * MR + i] * Bc[k * NR + j]

    # Solo escribe las filas/columnas reales — evita escribir fuera de C
    for i from 0 <= i < mr_real:
        for j from 0 <= j < nr_real:
            C[i * ldc + j] += ab[i][j]
### Fin de bloque generado por IA ############
### Micro-kernel #####################





### GEMM #############################################################################
'''
Orquestador del movimiento de datos desde la memoria principal hacia el procesador.
    Bc al caché L3 y en Ac hacia la L2 → segmentamos Br hacia la L1 → bloques fijos mr*nr en registros hardware

Todos los bloques de borde (donde M, N o K no son múltiplos exactos de mc, nc, kc) se calculan con imin() antes de continuar con las funciones del pipeline.

ANOTACION: La IA ha adaptado la función al caso de micro-kernel-edge. Es decir,
la ramificación if-else del bucle i_r principalmente, asi como corrección del calculo de
los valores "*_real" en cada bucle
'''

def gemm_forward_cython(float[:, ::1] A, float[:, ::1] B, float[:, ::1] C):

    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]

    ## CPU: Ryzen 5 3400G (w/avx2) ##
    # L1d 32  KB  core8-way       64 B
    # L2  512 KB  core8-way       64 B
    # L3  4   MB  shared16-way    64 B
    ###
    cdef int mr = MR
    cdef int nr = NR
    cdef int kc = 256   # Pasos de 32 - kc * 16 * 4 < L1d
    cdef int mc = 384   # mc * kc * 4               < L2
    cdef int nc = 2880  # nc * kc * 4               < L3
    #################################

    cdef int j_c, k_c, i_c, j_r, i_r
    cdef int mc_real, nc_real, kc_real, mr_real, nr_real
    cdef float* Bc_local = NULL
    cdef float* Ac_local = NULL

    with nogil:

        # Iteramos sobre columnas de B, paralelizando con OpenMP.
        # Cada hilo procesa un bloque distinto y necesita su propio buffer temporal Bc​,
        # que está dimensionado para almacenarse en la caché L3
        for j_c in prange(0, N, nc, schedule='static'):

            # nc_real es el ancho real de este bloque de columnas.
            # Para todos los bloques interiores nc_real == nc.
            # Para el bloque de borde derecho nc_real = N - j_c < nc.
            nc_real = imin(nc, N - j_c)

            Bc_local = <float*> malloc(kc * nc * sizeof(float))
            Ac_local = <float*> malloc(mc * kc * sizeof(float))

            # Realizamos el paso a lo largo de la dimensión común K y llamamos a pack_B
            for k_c from 0 <= k_c < K by kc:
                kc_real = imin(kc, K - k_c)

                # pack_B siempre recibe nc/nr completos para mantener el layout del buffer
                # los slots de borde se rellenan con 0 dentro de pack_B
                pack_B(&B[k_c, j_c], Bc_local, N, kc_real, nc, nr, nc_real if nc_real < nr else nr)

                # Descomponemos A en pasos "mc", cada iteración del bucle interior accede a un
                # bloque de Ac​ (mc​×kc​) que quepa en la L2 (512K)
                for i_c from 0 <= i_c < M by mc:
                    mc_real = imin(mc, M - i_c)

                    pack_A(&A[i_c, k_c], Ac_local, K, mc, kc_real, mr, mc_real if mc_real < mr else mr)

                    # Todas las iteraciones de este bucle acceden al mismo bloque de A, pero recorren
                    # micro-paneles de B que se ajustan al tamaño de caché L1 (32K x nucleo)
                    for j_r from 0 <= j_r < nc_real by nr:
                        nr_real = imin(nr, nc_real - j_r)

                        # Llamamos iterativamente al micro-kernel. Al tener empaquetados los datos
                        # el micro-kernel lee las secuencias correspondientes de Ac​ y Bc​ linealmente
                        # y acumula los resultados en C
                        for i_r from 0 <= i_r < mc_real by mr:
                            mr_real = imin(mr, mc_real - i_r)

                            if mr_real == MR and nr_real == NR:
                                micro_kernel(
                                    &Ac_local[i_r * kc_real],
                                    &Bc_local[j_r * kc_real],
                                    &C[i_c + i_r, j_c + j_r],
                                    N, kc_real
                                )
                            else:
                                micro_kernel_edge(
                                    &Ac_local[i_r * kc_real],
                                    &Bc_local[j_r * kc_real],
                                    &C[i_c + i_r, j_c + j_r],
                                    N, kc_real,
                                    mr_real, nr_real
                                )

            free(Ac_local)
            free(Bc_local)
################################################################################
