#!/bin/bash

# Compile, build, and run the homework.
export CC=gcc
cmake3 ../labs
make
./run_hw4.slurm

# Check that we have exactly four parameters.
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <INITIAL_DATASET_INDEX> <FINAL_DATASET_INDEX> <RUN_NUM> <NEW_KERNEL_BASE_NAME>"
    exit 1
fi

INITIAL_DATASET_INDEX=$1
FINAL_DATASET_INDEX=$2
RUN_NUM=$3
NEW_KERNEL_BASE_NAME=$4

# base_path is assumed to be the current folder.
base_path="."

# The regular expression to match the compute time (note: parentheses are escaped).
regex="Total compute time \\(ms\\) ([0-9.]+)"

# Create a folder for the results (if it doesn't already exist).
results_folder="../results"
mkdir -p "$results_folder"

# Loop over each dataset.
for (( dataset=INITIAL_DATASET_INDEX; dataset<FINAL_DATASET_INDEX; dataset++ ))
do
    echo "Processing dataset $dataset..."
    sum=0
    count=0

    # Loop over each run for the dataset.
    for (( run=1; run<=RUN_NUM; run++ ))
    do
        file_path="${base_path}/Histogram_output/output${dataset}_run${run}.txt"
        if [ ! -f "$file_path" ]; then
            echo "File $file_path not found. Skipping this run." >&2
            continue
        fi

        # Extract the compute time line using grep and the regular expression.
        line=$(grep -E "$regex" "$file_path")
        if [ -n "$line" ]; then
            # Extract the numeric value using sed.
            time=$(echo "$line" | sed -E "s/.*Total compute time \\(ms\\) ([0-9.]+).*/\\1/")
            if [ -n "$time" ]; then
                sum=$(echo "$sum + $time" | bc -l)
                count=$((count+1))
            fi
        fi
    done

    if [ $count -gt 0 ]; then
        average=$(echo "$sum / $count" | bc -l)
        # Create the new kernel file name by appending the dataset id.
        new_kernel_file="${NEW_KERNEL_BASE_NAME}_${dataset}.cu"
        # Copy the kernel file to the results folder with the new name.
        cp ../labs/hw4/Histogram/kernel.cu "${results_folder}/kernels/${new_kernel_file}"
        # Append the dataset id, average time, and new kernel file name to the results file.
        echo "($dataset, $average, ${new_kernel_file})" >> "${results_folder}/results.txt"
    else
        echo "No valid times found for dataset $dataset." >&2
    fi
done

echo "Results appended to ${results_folder}/results.txt and kernel copies saved in ${results_folder}/"
