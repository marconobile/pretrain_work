#!/bin/bash

# Define an array of configuration files
CONFIG_FILES=(
"./config/opioid_sequential_exp/opioid_0.yaml"
"./config/opioid_sequential_exp/opioid_1.yaml"
"./config/opioid_sequential_exp/opioid_2.yaml"
"./config/opioid_sequential_exp/opioid_3.yaml"
"./config/opioid_sequential_exp/opioid_4.yaml"
"./config/opioid_sequential_exp/opioid_5.yaml"
"./config/opioid_sequential_exp/opioid_6.yaml"
"./config/opioid_sequential_exp/opioid_7.yaml"
"./config/opioid_sequential_exp/opioid_8.yaml"
"./config/opioid_sequential_exp/opioid_9.yaml"
)

N=${#CONFIG_FILES[@]}
echo "Running $N experiments"

# Loop over each configuration file
for ((i=0; i<N; i++))
do
  # Get the current configuration file
  CONFIG_FILE=${CONFIG_FILES[$i]}
  echo "Running: $CONFIG_FILE"
  # Extract the prefix for the output file from the configuration file name
  OUTPUT_PREFIX=$(basename "$CONFIG_FILE" .yaml)
  OUTPUT_FILE="${OUTPUT_PREFIX}_out_${i}.txt"
  echo "Writing in: $OUTPUT_FILE"

  # Run geqtrain-train command
  geqtrain-train "$CONFIG_FILE"

  # Run eval.py script and write the output to the output file
  python ./source/scripts/eval.py -td "/home/nobilm@usi.ch/pretrain_paper/results/opioid/opioid_${i}" -d cuda:0 -bs 16 >> "$OUTPUT_FILE"
done
