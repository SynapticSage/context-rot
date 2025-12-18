#!/bin/bash
# Helper functions for Context Rot Research Workflow
# Created: 2025-11-28

# ============================================================================
# LOGGING
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================================================
# RATE LIMITS & DEPLOYMENT
# ============================================================================

detect_rate_limit() {
    local deployment_mode=$1

    if [ -n "$MAX_TOKENS_PER_MINUTE" ]; then
        echo "$MAX_TOKENS_PER_MINUTE"
        return
    fi

    case "$deployment_mode" in
        local)
            echo "50000"
            ;;
        cloud)
            echo "200000"
            ;;
        *)
            echo "100000"
            ;;
    esac
}

validate_deployment() {
    log "Validating deployment configuration..."

    if [[ "$DEPLOY_20B" != "local" && "$DEPLOY_20B" != "cloud" ]]; then
        log "ERROR: DEPLOY_20B must be 'local' or 'cloud', got: $DEPLOY_20B"
        exit 1
    fi

    if [[ "$DEPLOY_120B" != "local" && "$DEPLOY_120B" != "cloud" ]]; then
        log "ERROR: DEPLOY_120B must be 'local' or 'cloud', got: $DEPLOY_120B"
        exit 1
    fi

    local rate_20b=$(detect_rate_limit "$DEPLOY_20B")
    local rate_120b=$(detect_rate_limit "$DEPLOY_120B")

    log "Deployment configuration:"
    log "  20b model: $DEPLOY_20B deployment (rate limit: $rate_20b tokens/min)"
    log "  120b model: $DEPLOY_120B deployment (rate limit: $rate_120b tokens/min)"

    if [[ "$DEPLOY_20B" == "local" ]] || [[ "$DEPLOY_120B" == "local" ]]; then
        if [ -z "${GPT_OSS_BASE_URL:-}" ]; then
            log "WARNING: Local deployment selected but GPT_OSS_BASE_URL not set in .env"
            log "Make sure your local LLM server is running and accessible."
        fi
    fi

    if [[ "$DEPLOY_20B" == "cloud" ]] || [[ "$DEPLOY_120B" == "cloud" ]]; then
        if [ -z "${OPENROUTER_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
            log "WARNING: Cloud deployment selected but neither OPENROUTER_API_KEY nor OPENAI_API_KEY set in .env"
            log "Cloud inference will fail without valid API credentials."
        fi
    fi

    if [ ${RUN_20B} -eq 1 ]; then
        RATE_LIMIT_20B="$rate_20b"
    fi
    if [ ${RUN_120B} -eq 1 ]; then
        RATE_LIMIT_120B="$rate_120b"
    fi
}

# ============================================================================
# PREREQUISITES & DATASETS
# ============================================================================

check_prerequisites() {
    log "Checking prerequisites..."

    if [ ! -f ".env" ]; then
        log "ERROR: .env file not found. Please configure your API keys."
        log "Run: cp .env.example .env && nano .env"
        exit 1
    fi

    if [ -z "${VIRTUAL_ENV:-}" ]; then
        log "WARNING: Virtual environment not activated. Attempting to activate..."
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            log "Virtual environment activated."
        else
            log "ERROR: Virtual environment not found. Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
            exit 1
        fi
    fi

    mkdir -p "${RESULTS_DIR}"

    log "Prerequisites check complete."
}

check_datasets() {
    log "Checking required datasets..."

    local missing_datasets=0

    if [ ${RUN_NIAH} -eq 1 ]; then
        if [ ! -d "${PG_ESSAYS_DIR}" ]; then
            log "WARNING: PaulGrahamEssays dataset not found at ${PG_ESSAYS_DIR}"
            log "Please download from: https://drive.google.com/drive/folders/14uHYF65yu7cNGANungZX1NRboqwHHuVB?usp=sharing"
            missing_datasets=1
        fi
    fi

    if [ ${RUN_LONGMEMEVAL} -eq 1 ]; then
        if [ ! -f "${DATA_DIR}/cleaned_longmemeval_s_focused.csv" ] || [ ! -f "${DATA_DIR}/cleaned_longmemeval_s_full.csv" ]; then
            log "WARNING: LongMemEval datasets not found in ${DATA_DIR}"
            log "Please download from: https://drive.google.com/drive/folders/1AS1oytdcCH3y6p-DNuaNYI7I48IFgbfe?usp=sharing"
            missing_datasets=1
        fi
    fi

    if [ ${missing_datasets} -eq 1 ]; then
        log "ERROR: Missing required datasets. See docs/DATASETS.md for details."
        log "You can download datasets selectively based on which experiments you want to run."
        exit 1
    fi

    log "Dataset check complete."
}
