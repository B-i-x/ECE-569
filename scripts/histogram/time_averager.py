import os
import re
import numpy as np
import sys

# Usage example: python script.py outputs/histogram/v2/4/
if len(sys.argv) != 2:
    print("Usage: python script.py <base_path>")
    sys.exit(1)

base_path = sys.argv[1]

# Define your parameters here
INITIAL_DATASET_INDEX = 0
FINAL_DATASET_INDEX = 1
RUN_NUM = 10

# Regular expression to match the compute time line
compute_time_regex = re.compile(r"Total compute time \(ms\) ([0-9\.]+)")

# Store averaged times per dataset
dataset_avg_times = []

for dataset in range(INITIAL_DATASET_INDEX, FINAL_DATASET_INDEX):
    print(f"Processing dataset {dataset}...")
    times = []
    for run in range(1, RUN_NUM + 1):
        file_path = os.path.join(base_path, f"output{dataset}_run{run}.txt")
        with open(file_path, 'r') as file:
            for line in file:
                match = compute_time_regex.search(line)
                if match:
                    times.append(float(match.group(1)))
                    break

    # Calculate and store average
    average_time = np.mean(times)
    dataset_avg_times.append((dataset, average_time))

# Print averaged times
for dataset, avg_time in dataset_avg_times:
    print(f"Dataset {dataset}: Average Compute Time = {avg_time:.6f} ms")