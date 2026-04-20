### Archivo modificado para añadir soporte a los distintos algoritmos implementados.

from modules.conv2d import Conv2D
from modules.relu import ReLU
from modules.maxpool2d import MaxPool2D
from modules.flatten import Flatten
from modules.dense import Dense
from modules.softmax import Softmax
from models.basemodel import BaseModel
from modules.batchnorm import BatchNorm2D
from modules.dropout import Dropout


class OIANET_CIFAR100(BaseModel):
    def __init__(self, conv_algo=0, matmul_algo=0, maxpool2d_algo=0):
        print("Building OIANet for CIFAR-100")
        layers = [
            Conv2D(3, 32, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo),
            BatchNorm2D(32),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2, maxpool2d_algo=maxpool2d_algo),

            Conv2D(32, 64, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo),
            BatchNorm2D(64),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2, maxpool2d_algo=maxpool2d_algo),

            Conv2D(64, 128, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo),
            BatchNorm2D(128),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2, maxpool2d_algo=maxpool2d_algo),

            Flatten(),
            Dense(128 * 4 * 4, 256, matmul_algo=matmul_algo),
            ReLU(),
            Dropout(0.5),  # Dropout layer with 50% probability

            Dense(256, 100, matmul_algo=matmul_algo),
            Softmax()
        ]
        super().__init__(layers)