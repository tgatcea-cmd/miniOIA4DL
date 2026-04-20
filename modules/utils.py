import numpy as np


### MATMULS #################################
'''
Tecnicamente más sencillo que GEMM.
'''

### MATMUL Naive #######################
# Original
def _matmul_biases_naive(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    
    return C
##########################



### MATMUL Inline #######################
# Utilizamos la función matmul de numpy
def _matmul_biases_inline(A, B, C, bias):
    np.matmul(A, B, out=C)
    C += bias

    return C
##########################



### MATMUL Blocking #################
# Aplicamos la tecnica de blocking en cache, reutilizando
# los valores correspondientes calculados para gemm.pyx (cython)
# Dividiremos las matrices grandes en sub-bloques que entran en 
# la caché L2 y L3
def _matmul_biases_blocking(A, B, C, bias, MC=384, KC=256, NC=2880):
    M, K = A.shape
    K_ref, N = B.shape
    
    for j in range(0, N, NC):
        for k in range(0, K, KC):
            for i in range(0, M, MC):
                ### Bloque generado por IA ####################
                # Multiplicamos los sub-bloques y acumulamos el resultado en C
                C[i:i+MC, j:j+NC] += A[i:i+MC, k:k+KC] @ B[k:k+KC, j:j+NC]
                ### Bloque generado por IA ####################

    C += bias

    return C
####################



# ================================================= #
#PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biases(A, B, C, bias, matmul_algo=0):
    if matmul_algo == 0:
        return _matmul_biases_naive(A, B, C, bias)
    elif matmul_algo == 1:
        return _matmul_biases_inline(A, B, C, bias)
    elif matmul_algo == 2:
        return _matmul_biases_blocking(A, B, C, bias)
    else:
        raise ValueError("Error utils-matmul")
# ================================================= #

### Fin de MATMULS ##########################################