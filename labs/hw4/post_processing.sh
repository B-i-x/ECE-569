#!/bin/bash
# post_processing.sh
#
# Usage:
#   ./post_processing.sh <INITIAL_DATASET_INDEX> <FINAL_DATASET_INDEX> <RUN_NUM> <NEW_KERNEL_BASE_NAME> [DISCARD_RUNS]
#
# DISCARD_RUNS is an optional comma-separated list of run numbers to discard (e.g., "2,4").
#
# Example:
#   ./post_processing.sh 1 6 3 kernel_new 2,4

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <INITIAL_DATASET_INDEX> <FINAL_DATASET_INDEX> <RUN_NUM> <NEW_KERNEL_BASE_NAME> [DISCARD_RUNS]"
    exit 1
fi

INITIAL_DATASET_INDEX=$1
FINAL_DATASET_INDEX=$2
RUN_NUM=$3
NEW_KERNEL_BASE_NAME=$4
DISCARD_RUNS_STR=${5:-""}  # optional parameter; defaults to an empty string

# Convert the comma-separated list to an array.
IFS=',' read -r -a DISCARD_RUNS_ARRAY <<< "$DISCARD_RUNS_STR"

base_path="."  # Assumes current folder
results_folder="../results"
mkdir -p "$results_folder"

# Regular expression to match the compute time line.
regex="Total compute time \\(ms\\) ([0-9.]+)"

# Loop over each dataset.
for (( dataset=INITIAL_DATASET_INDEX; dataset<FINAL_DATASET_INDEX; dataset++ ))
do
    echo "Processing dataset $dataset..."
    sum=0
    count=0

    # Loop over each run for the dataset.
    for (( run=1; run<=RUN_NUM; run++ ))
    do
        # Check if the current run should be discarded.
        skip_run=false
        for discarded in "${DISCARD_RUNS_ARRAY[@]}"; do
            if [ "$run" -eq "$discarded" ]; then
                skip_run=true
                break
            fi
        done

        if [ "$skip_run" = true ]; then
            echo "Skipping run $run for dataset $dataset (discarded as outlier)."
            continue
        fi

        file_path="${base_path}/Histogram_output/output${dataset}_run${run}.txt"
        if [ ! -f "$file_path" ]; then
            echo "File $file_path not found. Skipping this run." >&2
            continue
        fi

        # Extract the compute time using grep and sed.
        line=$(grep -E "$regex" "$file_path")
        if [ -n "$line" ]; then
            time=$(echo "$line" | sed -E "s/.*Total compute time \\(ms\\) ([0-9.]+).*/\\1/")
            if [ -n "$time" ]; then
                sum=$(echo "$sum + $time" | bc -l)
                count=$((count+1))
            fi
        fi
    done

    if [ $count -gt 0 ]; then
        average=$(echo "$sum / $count" | bc -l)
        # Form the new kernel file name using the dataset id.
        new_kernel_file="${NEW_KERNEL_BASE_NAME}_${dataset}.cu"
        cp ../labs/hw4/Histogram/kernel.cu "${results_folder}/kernels/${new_kernel_file}"
        cp ../labs/hw4/Histogram/solution.cu "${results_folder}/solutions/solution_${new_kernel_file}"

        # Append dataset, average, and the new kernel file name to the results file.
        echo "($dataset, $average, ${new_kernel_file})" >> "${results_folder}/results.txt"
    else
        echo "No valid times found for dataset $dataset." >&2
    fi
done

echo "Post processing complete. Results appended to ${results_folder}/results.txt and kernel copies saved in ${results_folder}/"
