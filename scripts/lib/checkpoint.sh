#!/bin/bash
# Checkpoint and state management for Context Rot Research Workflow
# Created: 2025-11-28

# ============================================================================
# STATE FILE OPERATIONS
# ============================================================================

load_workflow_state() {
    if [ ! -f "$STATE_FILE" ]; then
        # Ensure directory exists before creating state file
        mkdir -p "$(dirname "$STATE_FILE")" 2>/dev/null || true
        echo '{}' > "$STATE_FILE" 2>/dev/null || true
    fi
}

save_workflow_state() {
    :
}

mark_step_complete() {
    local experiment=$1
    local model=$2
    local step=$3

    load_workflow_state

    python3 -c "
import json
import sys

try:
    with open('$STATE_FILE', 'r') as f:
        state = json.load(f)
except:
    state = {}

if '$experiment' not in state:
    state['$experiment'] = {}
if '$model' not in state['$experiment']:
    state['$experiment']['$model'] = {}

state['$experiment']['$model']['$step'] = True

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
" 2>/dev/null || true
}

is_step_complete() {
    local experiment=$1
    local model=$2
    local step=$3

    if [ ! -f "$STATE_FILE" ]; then
        return 1
    fi

    python3 -c "
import json
import sys

try:
    with open('$STATE_FILE', 'r') as f:
        state = json.load(f)

    result = state.get('$experiment', {}).get('$model', {}).get('$step', False)
    sys.exit(0 if result else 1)
except:
    sys.exit(1)
" 2>/dev/null

    return $?
}

# ============================================================================
# CSV VALIDATION
# ============================================================================

is_csv_complete() {
    local csv_path=$1
    local output_column=$2

    if [ ! -f "$csv_path" ]; then
        return 1
    fi

    python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$csv_path')

    if '$output_column' not in df.columns:
        sys.exit(1)

    if df['$output_column'].isna().any():
        sys.exit(1)

    if df['$output_column'].astype(str).str.startswith('ERROR').any():
        sys.exit(1)

    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null

    return $?
}

count_csv_rows() {
    local csv_path=$1
    local output_column=$2

    if [ ! -f "$csv_path" ]; then
        echo "0/0"
        return
    fi

    python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$csv_path')
    total = len(df)

    if '$output_column' not in df.columns:
        print(f'0/{total}')
        sys.exit(0)

    completed = (~df['$output_column'].isna() &
                 ~df['$output_column'].astype(str).str.startswith('ERROR')).sum()

    print(f'{completed}/{total}')
except:
    print('0/0')
" 2>/dev/null
}

# ============================================================================
# STEP RUNNER
# ============================================================================

run_step() {
    local experiment=$1
    local model=$2
    local step=$3
    local output_file=$4
    local output_column=$5
    shift 5

    if is_step_complete "$experiment" "$model" "$step"; then
        log "Skipping $experiment $model $step - already marked complete"
        return 0
    fi

    if [ -n "$output_file" ] && [ -n "$output_column" ]; then
        if is_csv_complete "$output_file" "$output_column"; then
            log "Skipping $experiment $model $step - output complete ($(count_csv_rows "$output_file" "$output_column"))"
            mark_step_complete "$experiment" "$model" "$step"
            return 0
        fi
    fi

    log "Running $experiment $model $step..."
    if "$@"; then
        mark_step_complete "$experiment" "$model" "$step"
        log "✓ $experiment $model $step complete"

        if [ "$step" == "inference" ] && [ -n "$output_file" ]; then
            local basename=$(basename "$output_file" .csv)
            local summary_file="${RESULTS_DIR}/${basename}_token_summary.json"
            show_final_token_summary "$summary_file"
        fi

        return 0
    else
        log "ERROR: $experiment $model $step failed"
        return 1
    fi
}
