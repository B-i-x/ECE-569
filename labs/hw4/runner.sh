#!/bin/bash

export CC=gcc
cmake3 ../labs
make
./run_hw4.slurm

