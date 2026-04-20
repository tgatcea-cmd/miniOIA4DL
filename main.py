### Cython Handler ##############################################
'''
Primeramente, comprobamos si existe algún fichero cython outdateado para
compilar. En caso contrario, continuamos ejecución.
'''
import subprocess
import pathlib
import sys

### Bloque generado por IA ##################
cython_path = pathlib.Path("cython_modules")
if cython_path.is_dir() and any(cython_path.glob("*.pyx")):
    pyx_files = list(cython_path.glob("*.pyx"))
    needs_compile = False
    
    for pyx in pyx_files:
        so_files = list(cython_path.glob(f"{pyx.stem}*.so"))
        if not so_files:
            needs_compile = True
            break
        
        pyx_mtime = pyx.stat().st_mtime
        if any(so.stat().st_mtime < pyx_mtime for so in so_files):
            needs_compile = True
            break

    if needs_compile:
        print("Cython modules modified or missing. Compiling...")
        try:
            subprocess.run(
                ["rm", "-rf", "build/", "*.c", "*.so"],
                cwd=cython_path,
                check=True
            )

            subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=cython_path,
                check=True
            )
            print("Cython compiling succesfull.")
        except subprocess.CalledProcessError as e:
            print(f"Error during compiling phase. {e}")
    else:
        print("Cython modules up to date. Skipping compilation.")
else:
    print("No cython module found to compile.")

print("\nContinuing execution...\n"+'#'*40+"\n\n")
### Fin de bloque generado por IA ##################
#################################################



from data.cifar100 import load_cifar100, normalize_images, one_hot_encode
from models.alexnet_cifar_100 import *
from models.resnet18_cifar_100 import ResNet18_CIFAR100
from models.tinycnn_cifar_100 import *
from models.oianet_cifar100 import OIANET_CIFAR100
from train import train
from eval import evaluate
from performance import perf
from data.cifar100_augmentator import CIFAR100Augmentor



def main(model_name, batch_size, epochs, learning_rate, conv_algo, performance, eval_only, matmul_algo, maxpool2d_algo):

    # !!Asegurarse de la ruta del dataset
    (train_images, train_labels), (test_images, test_labels) = load_cifar100(data_dir='./data/cifar-100-python')

    # NO TOCAR NADA DE AQUÍ PARA ABAJO
    train_images, test_images = normalize_images(train_images,test_images)
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    augmentor = CIFAR100Augmentor(crop_padding=4, flip_prob=0.5, noise_std=0.01)
    # Build and train model
    if model_name == 'AlexNet':
        model = AlexNet_CIFAR100(conv_algo=conv_algo)
    elif model_name == 'TinyCNN':
        model = TinyCNN(conv_algo=conv_algo)
    elif model_name == 'OIANet':
        model = OIANET_CIFAR100(conv_algo=conv_algo, matmul_algo=matmul_algo, maxpool2d_algo=maxpool2d_algo)
    else:
        model = ResNet18_CIFAR100(conv_algo=conv_algo)

    # Solamente se va a utilizar esta función para medir el rendimiento
    if performance:
        print("Measuring performance...")
        perf(model, train_images, train_labels, batch_size=batch_size)
    else: 
        if eval_only == False:
            train(model, train_images, train_labels, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
              save_path=f'saved_models/{model_name}', resume=True, test_images=test_images, test_labels=test_labels, augmentor=augmentor)
        else:
            _,_ = evaluate(model, test_images, test_labels, save_path=f'saved_models/{model_name}')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train a CNN model on CIFAR-100.')
    parser.add_argument('--model', type=str, choices=['AlexNet', 'TinyCNN', 'OIANet', 'ResNet18'], default='OIANet',
                        help='Model to train (default: OIANet)')
    parser.add_argument('--batch_size', type=int, default=176, help='Batch size for training (default: 176)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training (default: 0.01)')
    parser.add_argument('--performance', action='store_true', help='Enable performance measurement')
    parser.add_argument('--eval_only', action='store_true', help='Enable evaluation-only mode')
    
    ### Selección de Algoritmos #####################
    parser.add_argument('--conv_algo', type=int, default=3, choices=[0,1,2,3], help='Conv2d algorithm 0-direct, 1-im2col_gemm, 2-im2col_GEMM_vectorization, 3-im2col_GEMM_cython')
    parser.add_argument('--matmul_algo', type=int, default=2, choices=[0,1,2,3], help='Matmul algorithm 0-Naive, 1-Inline, 2-Blocking')
    parser.add_argument('--maxpool2d_algo', type=int, default=2, choices=[0,1,2,3], help='MaxPool2D algorithm 0-Naive, 1-Vectorized, 2-Cython')
    ########################


    args = parser.parse_args()
    
    model_name = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    performance = True # FOR OIANET performance
    conv_algo = args.conv_algo # PISTA: esto sirve para seleccionar nuevos algoritmos de convolucion
    eval_only = False # FOR OIANET performance

    matmul_algo = args.matmul_algo
    maxpool2d_algo = args.maxpool2d_algo
    
    main(model_name, batch_size, epochs, learning_rate, conv_algo, performance, eval_only, matmul_algo, maxpool2d_algo)
