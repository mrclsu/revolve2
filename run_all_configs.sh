#!/usr/bin/env bash

# Script to run all six configs safely, one after another
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
logs_dir="logs/run_all_configs_$timestamp"
mkdir -p "$logs_dir"

# Array to track results
declare -a results=()
declare -a config_names=("config1" "config2" "config3" "config4" "config5" "config6")
declare -a log_files=()

echo "Starting execution of all configs..."
echo "Logs will be saved to: $logs_dir"
echo "======================================="

# Run each config with error handling
for i in {1..6}; do
    config_name="${config_names[$((i-1))]}"
    log_file="$logs_dir/${config_name}_output.log"
    log_files+=("$log_file")
    
    echo ""
    echo "Running $config_name (config $i)..."
    echo "Output will be saved to: $log_file"
    echo "-----------------------------------"
    
    # Get start time
    start_time=$(date)
    
    # Run the config and capture both stdout and stderr to log file, while also showing on console
    echo "=== Config $i ($config_name) started at $start_time ===" > "$log_file"
    
    # Temporarily disable exit on error for individual config runs
    set +e
    mjpython project_2/main.py --config $i 2>&1 | tee -a "$log_file"
    exit_code=${PIPESTATUS[0]}  # Get the exit code of the first command in the pipeline
    set -e
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $config_name completed successfully at $(date)"
        echo "=== Config $i ($config_name) completed successfully at $(date) ===" >> "$log_file"
        results+=("$config_name: SUCCESS")
    else
        echo "❌ $config_name failed at $(date) with exit code $exit_code"
        echo "=== Config $i ($config_name) failed at $(date) with exit code $exit_code ===" >> "$log_file"
        results+=("$config_name: FAILED (exit code $exit_code)")
    fi
    
    echo "Started: $start_time"
    echo "Finished: $(date)"
    echo "Log saved to: $log_file"
done

echo ""
echo "======================================="
echo "Summary of all config runs:"
echo "======================================="

# Print results summary with log file paths
for i in "${!results[@]}"; do
    echo "${results[$i]} - Log: ${log_files[$i]}"
done

echo ""
echo "All configs execution completed at $(date)"
echo "All logs saved in directory: $logs_dir"

# Exit with error code if any config failed
failed_count=0
for result in "${results[@]}"; do
    if [[ $result == *"FAILED"* ]]; then
        ((failed_count++))
    fi
done

if [ $failed_count -gt 0 ]; then
    echo "⚠️  $failed_count config(s) failed"
    exit 1
else
    echo "🎉 All configs completed successfully!"
    exit 0
fi 