#!/bin/bash

cd ../..

OUTPUT_FOLDER="resources/inertial/"

# Set the range of heights, number of replicas, and zones with corresponding seeds
START_HEIGHT=10
END_HEIGHT=10
declare -A ZONE_SEEDS=( ["S1"]=510111393 ["S2"]=1383582242 ["S3"]=1016893250 ["S4"]=590537778 )  # Zone-seed mapping

# Loop through the range of max_height values and zones
for max_height in $(seq $START_HEIGHT $END_HEIGHT); do
    for zone in "${!ZONE_SEEDS[@]}"; do
        seed=${ZONE_SEEDS[$zone]}
        current_output_folder="${OUTPUT_FOLDER}${zone}/"
        echo "Running experiments for max_height = $max_height, zone = $zone with seed = $seed"
          python3 src/inertial_generate_certified_tree.py \
              --max_height "$max_height" \
              --initial_height 4 \
              --initial_points 60 \
              --points 20 \
              --zone "$zone" \
              --seed "$seed" \
              --output "$current_output_folder"
    done
done