#!/bin/bash

cd ../..

OUTPUT_FOLDER="resources/inertial/"

# Set the range of heights, number of replicas, and zones
START_HEIGHT=10
END_HEIGHT=10
REPLICAS=50  # Number of replicas per experiment
ZONES=("S1" "S2" "S3" "S4")  # Array of zone values

# Loop through the range of max_height values and zones
for max_height in $(seq $START_HEIGHT $END_HEIGHT); do
    for zone in "${ZONES[@]}"; do
        current_output_folder="${OUTPUT_FOLDER}${zone}/"
        echo "Running experiments for max_height = $max_height, zone = $zone"
        for replica_num in $(seq 1 $REPLICAS); do
            echo "  Replica $replica_num for max_height = $max_height, zone = $zone"
            python3 src/inertial_generate_certified_tree.py \
                --max_height "$max_height" \
                --initial_height 4 \
                --initial_points 60 \
                --points 20 \
                --zone "$zone" \
                --output "$current_output_folder"
        done
    done
done