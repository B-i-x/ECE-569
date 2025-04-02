#!/bin/bash

export CC=gcc
cmake3 ../labs
make
./run_hw4.slurm

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <INITIAL_DATASET_INDEX> <FINAL_DATASET_INDEX> <RUN_NUM>"
    exit 1
fi

INITIAL_DATASET_INDEX=$1
FINAL_DATASET_INDEX=$2
RUN_NUM=$3

# base_path is assumed to be the current folder
base_path="."

# The regular expression (parentheses are escaped)
regex="Total compute time \\(ms\\) ([0-9.]+)"

for (( dataset=INITIAL_DATASET_INDEX; dataset<FINAL_DATASET_INDEX; dataset++ ))
do
    echo "Processing dataset $dataset..."
    sum=0
    count=0

    for (( run=1; run<=RUN_NUM; run++ ))
    do
        file_path="${base_path}/output${dataset}_run${run}.txt"
        if [ ! -f "$file_path" ]; then
            echo "File $file_path not found. Skipping this run." >&2
            continue
        fi

        # Extract the compute time line using grep and the regular expression
        line=$(grep -E "$regex" "$file_path")
        if [ -n "$line" ]; then
            # Use sed to extract the numeric value from the matched line.
            time=$(echo "$line" | sed -E "s/.*Total compute time \\(ms\\) ([0-9.]+).*/\\1/")
            if [ -n "$time" ]; then
                # Add the extracted time to the sum (using bc for floating-point arithmetic)
                sum=$(echo "$sum + $time" | bc -l)
                count=$((count+1))
            fi
        fi
    done

    if [ $count -gt 0 ]; then
        average=$(echo "$sum / $count" | bc -l)
        echo "($dataset, $average)"
    else
        echo "No valid times found for dataset $dataset." >&2
    fi
done
