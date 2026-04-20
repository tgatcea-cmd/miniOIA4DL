### Este archivo ha sido modificado por IA para añadir las opciones de algoritmos correspondientes ###

from modules.dense import Dense
import numpy as np

def test_dense_forward_large():
    np.random.seed(42)

    batch_size = 4
    input_dim = 128
    output_dim = 64

    # Create dummy input
    input_data = np.random.randn(batch_size, input_dim).astype(np.float32)

    # Set known weights and bias
    weight = np.random.randn(input_dim, output_dim).astype(np.float32)
    bias = np.random.randn(output_dim).astype(np.float32)

    # Test each algorithm
    algos = {0: "Naive", 1: "Inline", 2: "Blocking"}
    for algo_code, algo_name in algos.items():
        # Initialize Dense layer
        dense = Dense(in_features=input_dim, out_features=output_dim, matmul_algo=algo_code)
        dense.weights = weight.copy()
        dense.biases = bias.copy()

        # Forward pass
        output = dense.forward(input_data)
        expected_output = np.dot(input_data, weight) + bias

        assert np.allclose(output, expected_output, atol=1e-5), f"Dense {algo_name} mismatch!"
        print(f"✅ Dense {algo_name} test passed.")

# Run the test
test_dense_forward_large()
