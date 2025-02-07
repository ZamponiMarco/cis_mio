#!/bin/bash

cd ../..

OUTPUT_FOLDER="resources/obstacle/"

# Set the range of heights, number of replicas, and zones with corresponding seeds
START_HEIGHT=10
END_HEIGHT=10
declare -A ZONE_SEEDS=( ["S1"]=739458743 ["S2"]=1304740369 ["S3"]=452662493 ["S4"]=1977904097 )  # Zone-seed mapping

# Loop through the range of max_height values and zones
for max_height in $(seq $START_HEIGHT $END_HEIGHT); do
    for zone in "${!ZONE_SEEDS[@]}"; do
        seed=${ZONE_SEEDS[$zone]}
        current_output_folder="${OUTPUT_FOLDER}${zone}/"
        echo "Running experiments for max_height = $max_height, zone = $zone with seed = $seed"
          python3 src/obstacle_generate_certified_tree.py \
              --max_height "$max_height" \
              --initial_height 4 \
              --initial_points 60 \
              --points 20 \
              --zone "$zone" \
              --seed "$seed" \
              --output "$current_output_folder"
    done
done
