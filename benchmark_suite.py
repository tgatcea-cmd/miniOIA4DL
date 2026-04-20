### Benchmark Suite #########################################################
'''
Este suite ha sido creado para mejorar el testeo y comprobación de mejoras de
forma individual por capas. Es interactivo por linea de comandos. Aprovecha
los test unitarios de 'unit_tests/'
'''

### Bloque generado por IA ###############################################
import time
import numpy as np
import sys
import subprocess
import os
import csv
from datetime import datetime
from modules.conv2d import Conv2D
from modules.maxpool2d import MaxPool2D
from modules.dense import Dense
from modules.batchnorm import BatchNorm2D

def benchmark_module(name, module_gen, input_shape, iterations=10):
    print(f"\n--- Benchmarking {name} ---")
    
    # Warmup
    try:
        module = module_gen()
        x = np.random.randn(*input_shape).astype(np.float32)
        _ = module.forward(x, training=False)
        
        start = time.time()
        for _ in range(iterations):
            _ = module.forward(x, training=False)
        end = time.time()
        
        avg_time = (end - start) / iterations
        ips = 1.0 / avg_time if avg_time > 0 else 0
        print(f"Avg Time: {avg_time*1000:.3f}ms | IPS: {ips:.2f}")
        return avg_time
    except Exception as e:
        print(f"Error benchmarking {name}: {e}")
        return None

def run_full_benchmark():
    folder = "logs/full_benchmark"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Definición de todas las combinaciones a probar
    # Estructura: (Nombre, Generador, Input_Shape, Lista_Algos)
    configs = [
        ("Conv2D", lambda c: Conv2D(3, 32, 3, conv_algo=c), (16, 3, 32, 32), [0, 1, 2, 3]),
        ("MaxPool2D", lambda c: MaxPool2D(2, 2, maxpool2d_algo=c), (16, 32, 32, 32), [0, 1, 2]),
        ("Dense", lambda c: Dense(512, 1024, matmul_algo=c), (16, 512), [0, 1, 2]),
        ("BatchNorm2D", lambda c: BatchNorm2D(32), (16, 32, 32, 32), [0])
    ]

    print(f"\nStarting Exhaustive Benchmark...")
    print(f"Saving results to: {folder}/")

    for name, gen, shape, algos in configs:
        for algo in algos:
            avg_time = benchmark_module(f"{name}_Algo{algo}", lambda a=algo: gen(a), shape)
            if avg_time:
                # El nombre del archivo incluye metadatos clave para trazabilidad
                filename = f"{folder}/{name}_algo{algo}_batch{shape[0]}_{timestamp}.csv"
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "module", "algo", "batch_size", "input_shape", "avg_time_ms", "ips"])
                    writer.writerow([
                        timestamp, 
                        name, 
                        algo, 
                        shape[0], 
                        str(shape), 
                        avg_time * 1000, 
                        1.0 / avg_time
                    ])
    
    print(f"\n[FINISHED] All combinations tested. Metadata saved in filenames.")

def main():
    print("miniOIA4DL Interactive Benchmark Suite")
    print("======================================")
    
    while True:
        print("\nSelect action:")
        print("1. Conv2D (Algorithms: Direct, im2col, Vectorized, Cython)")
        print("2. MaxPool2D (Algorithms: Naive, Vectorized, Cython)")
        print("3. Dense (Algorithms: Naive, Inline, Blocking)")
        print("4. BatchNorm2D")
        print("5. FULL BENCHMARK (All combinations + CSV Export)")
        print("6. Run Unit Tests")
        print("7. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == '1':
            B, C, H, W = 16, 3, 32, 32
            out_c, k = 32, 3
            input_shape = (B, C, H, W)
            algos = {
                0: "Direct (Naive Loops)",
                1: "im2col + GEMM (NumPy)",
                2: "im2col + GEMM (Vectorized)",
                3: "im2col + GEMM (Cython)"
            }
            for code, name in algos.items():
                benchmark_module(f"Conv2D - {name}", 
                               lambda c=code: Conv2D(C, out_c, k, conv_algo=c),
                               input_shape)
                    
        elif choice == '2':
            B, C, H, W = 16, 32, 32, 32
            input_shape = (B, C, H, W)
            algos = {
                0: "Naive",
                1: "Vectorization",
                2: "Cython"
            }
            for code, name in algos.items():
                benchmark_module(f"MaxPool2D - {name}", 
                               lambda c=code: MaxPool2D(2, 2, maxpool2d_algo=c), 
                               input_shape)
            
        elif choice == '3':
            B, Din, Dout = 16, 512, 1024
            input_shape = (B, Din)
            algos = {
                0: "Naive",
                1: "Inline (NumPy)",
                2: "Blocking"
            }
            for code, name in algos.items():
                benchmark_module(f"Dense - {name}", 
                               lambda c=code: Dense(Din, Dout, matmul_algo=c), 
                               input_shape)
            
        elif choice == '4':
            B, C, H, W = 16, 32, 32, 32
            benchmark_module("BatchNorm2D", 
                           lambda: BatchNorm2D(C), 
                           (B, C, H, W))
            
        elif choice == '5':
            run_full_benchmark()

        elif choice == '6':
            print("\nRunning unit tests...")
            try:
                subprocess.run(["bash", "unit_tests/run.sh"], check=True)
            except Exception as e:
                print(f"Error running unit tests: {e}")

        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
### Fin de bloque generado por IA ###############################################
