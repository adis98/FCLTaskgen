#!/usr/bin/env bash

run_fl_experiment() {
    local alpha=$1
    local client_id=$2
    local replication=$3
    python3 -m run_fl --alpha $alpha --clientid $client_id --replication $replication > output/${alpha}_${client_id}_${replication}.txt 2>&1 &
}

alphas=(
    1
    10
    100
    1000
)

for a in "${alphas[@]}"; do
  echo "$(date) Running for alpha: ${alpha}"
  for ((r=0; r<10; r++)); do
    echo "$(date) Running for replication: ${r}"
    for ((i=-1; i<4; i++)); do
      run_fl_experiment $a $i $r
    done

    # Wait for all background processes to finish
    wait
    echo "All scripts have finished executing. Continuing..."
  done
done
# Iterate over the scripts and run them in the background



