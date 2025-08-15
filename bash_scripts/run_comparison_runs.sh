#!/bin/bash
# This script runs all experiments for the delta weights detection with the mnist dataset.

# 0. Source the virtual environment
source .venv/bin/activate

# 0.1 Get the dataset
if [ -z "$1" ]; then
    echo "Error: Please provide a dataset name (mnist, cifar10)"
    exit 1
fi
DATASET="$1"

# 1.1 get the attack names
ATTACKS_STRING=$(python get_attack_names.py)
readarray -t ATTACKS <<< "$ATTACKS_STRING"

# 1.2 number of clients
NUM_CLIENTS=(10 100)

# 1.3 percentages of free riders
PERCENT_MALICIOUS=(0.3)

# 1.4 run everything 3 times
RUN_3_TIMES=(1 2 3)

TOTAL_EXPERIMENTS=$((${#RUN_3_TIMES[@]} *  ${#ATTACKS[@]} * ${#NUM_CLIENTS[@]} * ${#PERCENT_MALICIOUS[@]} ))
CURRENT_EXPERIMENT=0
echo "Total experiments to run: $TOTAL_EXPERIMENTS"

START_TIME=$(date +%s)

# 2. run all experiments
for ATTACK in "${ATTACKS[@]}"; do
    # Run it with every attack
    for NUM_C in "${NUM_CLIENTS[@]}"; do
        # Run it with both 10 and 100 clients
        for PERC_M in "${PERCENT_MALICIOUS[@]}"; do
            # Run with 10, 30 and 70 percent
            for n in "${RUN_3_TIMES[@]}"; do
            # Run everything 3 times
                # Increment the counter
                CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
                # Print the progress
                # 4.2 Calculate and print the progress estimate
                ELAPSED_TIME=$(( $(date +%s) - START_TIME ))
                if [[ $CURRENT_EXPERIMENT -gt 0 ]]; then
                    AVG_TIME=$(( ELAPSED_TIME / CURRENT_EXPERIMENT ))
                    REMAINING_TIME=$(( AVG_TIME * (TOTAL_EXPERIMENTS - CURRENT_EXPERIMENT) ))
                    REMAINING_MINS=$(( REMAINING_TIME / 60 ))
                    REMAINING_SECS=$(( REMAINING_TIME % 60 ))
                    echo "Progress: $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS. Estimated remaining time: ${REMAINING_MINS}m ${REMAINING_SECS}s."
                fi

                # 2.1 generate one configuration
                python generate_config.py "$DATASET" $NUM_C $PERC_M "$ATTACK" "no_detection"

                # 2.2 run one experiment
                flwr run . local-sim-gpu
            done
        done
    done
done

echo "Finished the comparison run for $DATASET"