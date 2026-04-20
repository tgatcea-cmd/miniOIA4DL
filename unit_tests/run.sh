#!/bin/bash
folder="unit_tests/"
# Only run files starting with test_
utests=$(ls unit_tests/test_*.py)
### Bloque modificado por IA ########
# Modificado por IA para garantizar ejecución automatizada desde el benchmark_suite
# Move to root to ensure imports work
cd "$(dirname "$0")/.."
for test in $utests
do
    echo "Running $test..."
    PYTHONPATH=. python3 "$test"
done
### Fin de bloque modificado por IA #########
