#!/bin/bash
# run_slurm.sh
#
# This script sets the compiler, configures the build, compiles the project,
# and executes the slurm job.
git pull
export CC=gcc
cmake3 ../labs
make
./run_hw4.slurm