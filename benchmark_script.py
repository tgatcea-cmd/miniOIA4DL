### Benchmark runner iterativo(-n) -> CSV(-o) ###########################
'''
Este script está hecho para facilitar la comparación de datos de forma más estable, realiza N runs y calcula el resultado medio.
Luego, puede ser guardado en un csv custom.
'''



# ============================================================================================= #
my_cmd = ["python", "main.py", "--conv_algo", "3", "--maxpool2d_algo", "2", "--matmul_algo", "2", "--batch_size", "184"]
# ============================================================================================= #





### BLOQUE GENERADO CON IA #############################
import subprocess
import csv
import argparse

def run_benchmark(n_runs, output_csv):
    cmd = my_cmd
    
    # Store aggregated data: list of dicts to preserve order
    # Format: [{'layer': str, 'batch': str, 'time': float, 'perf': float}]
    aggregated = []
    
    for i in range(n_runs):
        print(f"Executing run {i+1}/{n_runs}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[!] FATAL: main.py crashed during run {i+1}.")
            print(f"--- TRACEBACK FROM MAIN.PY ---")
            print(e.stderr)
            print(f"------------------------------")
            exit(1)
        parsing = False
        row_idx = 0
        
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith("FW Layer;"):
                parsing = True
                continue
            if line.startswith("==="):
                parsing = False
                continue
                
            if parsing and line:
                parts = line.split(';')
                if len(parts) == 4:
                    layer, batch, t_str, p_str = parts
                    t, p = float(t_str), float(p_str)
                    
                    if i == 0:
                        # Initialize tracking structure on first run
                        aggregated.append({
                            'layer': layer, 
                            'batch': batch, 
                            'time': t, 
                            'perf': p
                        })
                    else:
                        # Accumulate
                        aggregated[row_idx]['time'] += t
                        aggregated[row_idx]['perf'] += p
                    
                    row_idx += 1

    # Variables to track total network performance
    total_avg_time = 0.0
    batch_size = 1

    # Calculate averages and write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["FW Layer", "Batch", "Avg Time(s)", "Avg Performance(imgs/s)"])

        for row in aggregated:
            avg_time = row['time'] / n_runs
            avg_perf = row['perf'] / n_runs
            writer.writerow([row['layer'], row['batch'], f"{avg_time:.4f}", f"{avg_perf:.2f}"])

            # Add to total time and grab the batch size
            total_avg_time += avg_time
            batch_size = int(row['batch'])
            
    print(f"Aggregation complete. Results saved to {output_csv}")

    # --- NEW: CALCULATE AND PRINT OVERALL IPS ---
    if total_avg_time > 0:
        total_ips = batch_size / total_avg_time
        print("\n" + "="*45)
        print(" NETWORK PERFORMANCE SUMMARY")
        print("="*45)
        print(f"  Avg Time Per Batch:    {total_avg_time:.4f} seconds")
        print(f"  Total Average IPS:     {total_ips:.2f} images/sec")
        print("="*45 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Averaging benchmark runs.")
    parser.add_argument("-n", "--runs", type=int, default=5, help="Number of times to execute.")
    parser.add_argument("-o", "--output", type=str, default="average_results.csv", help="Output CSV file.")
    args = parser.parse_args()
    
    run_benchmark(args.runs, args.output)
### FIN BLOQUE GENERADO CON IA #############################
