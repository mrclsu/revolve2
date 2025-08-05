#!/usr/bin/env bash

# Script to run a specific experiment config for n repetitions
# Based on mjpy script structure

set -e  # Exit on error for the script itself, but we'll handle individual config errors
set -o pipefail  # Make pipelines return the exit code of the last command that failed

# check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment activated, activating..."
    source .venv/bin/activate
fi

export DYLD_FRAMEWORK_PATH=/System/Library/Frameworks

# Parse command line arguments
usage() {
    echo "Usage: $0 --config <config_name> [--repetitions <n>] [--standard]"
    exit 1
}

config_name=""
num_repetitions=5  # Default value
use_standard=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
        config_name="$2"
        shift; shift
        ;;
        --repetitions)
        num_repetitions="$2"
        shift; shift
        ;;
        --standard)
        use_standard=1
        shift
        ;;
        *)
        usage
        ;;
    esac

done

if [ -z "$config_name" ]; then
    echo "Error: --config <config_name> is required."
    usage
fi

# Create logs directory with timestamp
timestamp=$(date '+%Y%m%d_%H%M%S')
logs_dir="logs/run_${config_name}_repetitions_$timestamp"
mkdir -p "$logs_dir"

# Array to track results
declare -a results=()
declare -a log_files=()

echo "Starting execution of $config_name for $num_repetitions repetitions..."
echo "Logs will be saved to: $logs_dir"
echo "======================================="

# Run the config for n repetitions with error handling
for ((i=1; i<=num_repetitions; i++)); do
    log_file="$logs_dir/${config_name}_run_${i}_output.log"
    log_files+=("$log_file")
    
    echo ""
    echo "Running $config_name (repetition $i/$num_repetitions)..."
    echo "Output will be saved to: $log_file"
    echo "-----------------------------------"
    
    # Get start time
    start_time=$(date)
    
    # Run the config and capture both stdout and stderr to log file, while also showing on console
    echo "=== ${config_name} repetition $i started at $start_time ===" > "$log_file"
    
    # Temporarily disable exit on error for individual config runs
    set +e
    if [ $use_standard -eq 1 ]; then
        python project_2/standard_main.py --config $config_name 2>&1 | tee -a "$log_file"
    else
        python project_2/main.py --config $config_name 2>&1 | tee -a "$log_file"
    fi
    exit_code=${PIPESTATUS[0]}  # Get the exit code of the first command in the pipeline
    set -e
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $config_name repetition $i completed successfully at $(date)"
        echo "=== ${config_name} repetition $i completed successfully at $(date) ===" >> "$log_file"
        results+=("$config_name repetition $i: SUCCESS")
    else
        echo "❌ $config_name repetition $i failed at $(date) with exit code $exit_code"
        echo "=== ${config_name} repetition $i failed at $(date) with exit code $exit_code ===" >> "$log_file"
        results+=("$config_name repetition $i: FAILED (exit code $exit_code)")
    fi
    
    echo "Started: $start_time"
    echo "Finished: $(date)"
    echo "Log saved to: $log_file"
done

echo ""
echo "======================================="
echo "Summary of all repetitions:"
echo "======================================="

# Print results summary with log file paths
for i in "${!results[@]}"; do
    echo "${results[$i]} - Log: ${log_files[$i]}"
done

echo ""
echo "All repetitions completed at $(date)"
echo "All logs saved in directory: $logs_dir"

# Exit with error code if any repetition failed
failed_count=0
for result in "${results[@]}"; do
    if [[ $result == *"FAILED"* ]]; then
        ((failed_count++))
    fi
    
done

if [ $failed_count -gt 0 ]; then
    echo "⚠️  $failed_count repetition(s) failed"
    exit 1
else
    echo "🎉 All repetitions completed successfully!"
    exit 0
fi