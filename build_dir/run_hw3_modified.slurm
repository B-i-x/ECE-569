#!/bin/bash

### Set the job name 
#SBATCH --job-name=devicequery

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=standard

#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:00:50

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules
###module load openmpi3
###module load cuda11/11.0

### set directory for job execution, replace the following path with your own path
hw_path=~aromerolozano/ece569/build_dir

cd $hw_path

# mkdir -p BasicMatrixMultiplication_output
# mkdir TiledMatrixMultiplication_output
###
PATHS[0]=$hw_path/MatrixMultiplication/Dataset/0;

#PATHS[0]=$hw_path/MatrixMultiplication/Dataset/10;
#PATHS[1]=$hw_path/MatrixMultiplication/Dataset/11;
#PATHS[2]=$hw_path/MatrixMultiplication/Dataset/12;
#PATHS[3]=$hw_path/MatrixMultiplication/Dataset/13;

count=0
for i in ${PATHS[@]}
do
   file=output$((count)).txt
   ./BasicMatrixMultiplication_Solution -e "$i/output.raw" -i "$i/input0.raw,$i/input1.raw" -t matrix > BasicMatrixMultiplication_output/"$file"
 count=$((count+1))
done

PATHS[0]=$hw_path/TiledMatrixMultiplication/Dataset/0;

#PATHS[0]=$hw_path/TiledMatrixMultiplication/Dataset/10;
#PATHS[1]=$hw_path/TiledMatrixMultiplication/Dataset/11;
#PATHS[2]=$hw_path/TiledMatrixMultiplication/Dataset/12;
#PATHS[3]=$hw_path/TiledMatrixMultiplication/Dataset/13;

TILEEXES[0]=$hw_path/no_timer_util_Solution_tile4;
TILEEXES[1]=$hw_path/no_timer_util_Solution_tile8;
TILEEXES[2]=$hw_path/no_timer_util_Solution_tile16;
TILEEXES[3]=$hw_path/no_timer_util_Solution_tile32;


# Loop over each tile-executable and each dataset path
exe_count=4
for exe in "${TILEEXES[@]}"; do
    count=0

    for dataset_path in "${PATHS[@]}"; do
        # Construct a unique output filename based on which executable and dataset
        out_file="output_${exe_count}_${count}.txt"

        # Run the executable with the dataset
        "$exe" -e "${dataset_path}/output.raw" -i "${dataset_path}/input0.raw,${dataset_path}/input1.raw" -t matrix > "TiledMatrixMultiplication_output/${out_file}"

        count=$((count+1))
    done
    exe_count=$((exe_count*2))
done
  



