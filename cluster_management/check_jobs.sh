#!/bin/bash

USER=$(whoami)

# Get job IDs
squeue -u "$USER" -h -o "%A" | while read -r job; do
    echo "Job ID: $job"

    # Get job info
    JOB_INFO=$(scontrol show job "$job" 2>/dev/null)

    if [[ -z "$JOB_INFO" ]]; then
        echo "Could not retrieve job info with scontrol. Skipping..."
        echo ""
        continue
    fi

    # Extract details using grep -oP
    JOB_STATE=$(echo "$JOB_INFO" | grep -oP 'JobState=\K\S+')
    JOB_NAME=$(echo "$JOB_INFO" | grep -oP 'JobName=\K\S+')
    OUTPUT_PATH=$(echo "$JOB_INFO" | grep -oP 'StdOut=\K\S+')
    REASON=$(echo "$JOB_INFO" | grep -oP 'Reason=\K\S+')

    echo "State: $JOB_STATE"
    echo "Name: $JOB_NAME"

    if [[ -n "$OUTPUT_PATH" ]]; then
        echo "Output path: $OUTPUT_PATH"
        JOB_DIR=$(dirname "$OUTPUT_PATH")
    else
        echo "Output path not set."
        JOB_DIR=""
    fi

    if [[ "$JOB_STATE" == "PENDING" ]]; then
        echo "Reason for pending: $REASON"
        echo ""
        continue
    fi

    # If job is running, check OSZICAR/INCAR
    if [[ -n "$JOB_DIR" && -f "$JOB_DIR/OSZICAR" && -f "$JOB_DIR/INCAR" ]]; then
        CURRENT_STEP=$(grep -E "^[[:space:]]*[0-9]+[[:space:]]+T=|^[[:space:]]*[0-9]+[[:space:]]+F=" "$JOB_DIR/OSZICAR" | tail -1 | awk '{print $1}')
        TOTAL_STEPS=$(grep -i "^[[:space:]]*NSW[[:space:]]*=" "$JOB_DIR/INCAR" | awk -F'=' '{print $2}' | awk '{print $1}')

        if [[ -n "$CURRENT_STEP" && -n "$TOTAL_STEPS" ]]; then
            PERCENT=$(awk -v cur="$CURRENT_STEP" -v total="$TOTAL_STEPS" 'BEGIN { printf "%.2f", (cur / total) * 100 }')
            echo "Completion: $CURRENT_STEP / $TOTAL_STEPS steps ($PERCENT%)"
        else
            echo "Could not parse step count or NSW from $JOB_DIR"
        fi
    else
        echo "Missing OSZICAR or INCAR in $JOB_DIR"
    fi

    echo ""
done
