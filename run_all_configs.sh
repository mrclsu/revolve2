#!/usr/bin/env bash

# Script to run all experiment configs safely, one after another  
# Based on mjpy script structure

set -e  # Exit on error for the script itself, but we'll handle individual config errors
set -o pipefail  # Make pipelines return the exit code of the last command that failed

# check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment activated, activating..."
    source .venv/bin/activate
fi

export DYLD_FRAMEWORK_PATH=/System/Library/Frameworks

# Create logs directory with timestamp
timestamp=$(date '+%Y%m%d_%H%M%S')
logs_dir="logs/run_all_experiments_$timestamp"
mkdir -p "$logs_dir"

# Array to track results
declare -a results=()
declare -a config_names=(
    "experiment_lowest_fitness_max_distance"
    # "experiment_lowest_fitness_head_stability"
    # "experiment_max_age_max_distance"
    # "experiment_max_age_head_stability"
)
declare -a log_files=()

echo "Starting execution of all experiment configs..."
echo "Logs will be saved to: $logs_dir"
echo "======================================="

# Run each config with error handling
for i in "${!config_names[@]}"; do
    config_name="${config_names[$i]}"
    log_file="$logs_dir/${config_name}_output.log"
    log_files+=("$log_file")
    
    echo ""
    echo "Running $config_name (experiment $((i+1))/4)..."
    echo "Output will be saved to: $log_file"
    echo "-----------------------------------"
    
    # Get start time
    start_time=$(date)
    
    # Run the config and capture both stdout and stderr to log file, while also showing on console
    echo "=== Experiment $((i+1)) ($config_name) started at $start_time ===" > "$log_file"
    
    # Temporarily disable exit on error for individual config runs
    set +e
    mjpython project_2/main.py --config $config_name 2>&1 | tee -a "$log_file"
    exit_code=${PIPESTATUS[0]}  # Get the exit code of the first command in the pipeline
    set -e
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $config_name completed successfully at $(date)"
        echo "=== Experiment $((i+1)) ($config_name) completed successfully at $(date) ===" >> "$log_file"
        results+=("$config_name: SUCCESS")
    else
        echo "❌ $config_name failed at $(date) with exit code $exit_code"
        echo "=== Experiment $((i+1)) ($config_name) failed at $(date) with exit code $exit_code ===" >> "$log_file"
        results+=("$config_name: FAILED (exit code $exit_code)")
    fi
    
    echo "Started: $start_time"
    echo "Finished: $(date)"
    echo "Log saved to: $log_file"
done

echo ""
echo "======================================="
echo "Summary of all experiment runs:"
echo "======================================="

# Print results summary with log file paths
for i in "${!results[@]}"; do
    echo "${results[$i]} - Log: ${log_files[$i]}"
done

echo ""
echo "All experiments execution completed at $(date)"
echo "All logs saved in directory: $logs_dir"

# Exit with error code if any experiment failed
failed_count=0
for result in "${results[@]}"; do
    if [[ $result == *"FAILED"* ]]; then
        ((failed_count++))
    fi
done

if [ $failed_count -gt 0 ]; then
    echo "⚠️  $failed_count experiment(s) failed"
    exit 1
else
    echo "🎉 All experiments completed successfully!"
    exit 0
fi 