#!/bin/bash

# Set the loss function you want to test here
loss_function="mmd_loss"  # Change this line for each experiment

mkdir -p results

for seed in $(seq 0 19); do
    echo "Running: loss=${loss_function}, seed=${seed}"
    python QMC_hail.py --seed $seed --loss $loss_function > results/${loss_function}_seed${seed}.log
done
