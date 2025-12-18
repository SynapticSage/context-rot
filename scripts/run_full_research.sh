#!/bin/bash
# Complete Context Rot Research Workflow for GPT-OSS Models
# Created: 2025-11-19
# Refactored: 2025-11-28 (modularized)
#
# IMPORTANT: Run this script from the repository root directory:
#   ./scripts/run_full_research.sh
#   ./scripts/run_full_research.sh -t  # Test mode

set -u  # Exit on undefined variable

# ============================================================================
# SETUP
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Source modular functions
source "${SCRIPT_DIR}/lib/helpers.sh"
source "${SCRIPT_DIR}/lib/checkpoint.sh"
source "${SCRIPT_DIR}/lib/display.sh"

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

TEST_MODE=0
TRUNCATE_MODE=0

usage() {
    echo "Usage: $0 [-t] [-T] [-h]"
    echo "  -t    Enable test mode (reduced samples, ~10-20 min runtime)"
    echo "  -T    Enable truncation (truncate oversized prompts to fit context limit)"
    echo "  -h    Show this help message"
    exit 0
}

while getopts "tTh" opt; do
    case $opt in
        t) TEST_MODE=1 ;;
        T) TRUNCATE_MODE=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations
# GPT-OSS models - the target models for this research
MODEL_20B="openai/gpt-oss-20b"             # OpenRouter: gpt-oss-20b (21B MoE, 3.6B active)
MODEL_120B="openai/gpt-oss-120b"           # OpenRouter: gpt-oss-120b (117B MoE, 5.1B active)
MAX_CONTEXT_LENGTH=131072

# Deployment modes: "local" or "cloud"
DEPLOY_20B="cloud"
DEPLOY_120B="cloud"

# Rate limits (auto-adjusted based on deployment mode)
MAX_TOKENS_PER_MINUTE=""

# Repeated Words experiment settings
COMMON_WORD="apple"
MODIFIED_WORD="apples"
MODEL_MAX_OUTPUT_TOKENS=32768

# LLM Judge model
JUDGE_MODEL="gpt-4o-mini-2024-07-18"

# Experiment selection
RUN_NIAH=1
RUN_LONGMEMEVAL=1
RUN_REPEATED_WORDS=1

# Model selection
RUN_20B=1
RUN_120B=1

# Paths
DATA_DIR="./data"
RESULTS_DIR="./results"
PG_ESSAYS_DIR="${DATA_DIR}/PaulGrahamEssays"
NIAH_PROMPTS_DIR="${DATA_DIR}/niah_prompts"
STATE_FILE="${RESULTS_DIR}/.workflow_state.json"

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

main() {
    log "=========================================="
    log "Context Rot Research - Full Workflow"
    log "=========================================="
    log "Models: ${MODEL_20B} (20b), ${MODEL_120B} (120b)"
    log "Experiments: NIAH=${RUN_NIAH}, LongMemEval=${RUN_LONGMEMEVAL}, Repeated Words=${RUN_REPEATED_WORDS}"
    log ""

    check_prerequisites
    validate_deployment
    check_datasets

    show_test_mode_banner
    show_cost_estimate
    show_token_tracking_banner

    load_workflow_state
    show_workflow_status
    log ""

    # Set test mode flag and file prefix for Python scripts
    if [ ${TEST_MODE} -eq 1 ]; then
        TEST_FLAG="--test-mode"
        FILE_PREFIX="test_"
    else
        TEST_FLAG=""
        FILE_PREFIX=""
    fi

    # Set truncation flag for oversized prompts
    if [ ${TRUNCATE_MODE} -eq 1 ]; then
        TRUNCATE_FLAG="--truncate-to-fit"
        log "Truncation enabled: oversized prompts will be truncated from front"
    else
        TRUNCATE_FLAG=""
    fi

    # ========================================================================
    # EXPERIMENT 1: NIAH Extension
    # ========================================================================

    if [ ${RUN_NIAH} -eq 1 ]; then
        log "=========================================="
        log "Starting NIAH Extension Experiment"
        log "=========================================="

        cd experiments/niah_extension

        # Generate haystacks (only if needed or in test mode)
        if [ ! -f "../../${NIAH_PROMPTS_DIR}/niah_prompts_sequential.csv" ] || [ ${TEST_MODE} -eq 1 ]; then
            log "Generating haystacks with semantic needles..."
            python run/create_haystacks.py \
                --haystack-folder "../../${PG_ESSAYS_DIR}" \
                --needle "It sometimes surprises people when I tell them I write every week. I was also surprised when my friend from my freshman year History course was doing the same thing, but looking back, I only wish I started earlier." \
                --question "What was the best writing advice I got from my college classmate?" \
                --output-folder "../../${NIAH_PROMPTS_DIR}" \
                ${TEST_FLAG}
            log "Haystacks generated successfully."
        else
            log "Haystacks already exist, skipping generation."
        fi

        # Run inference
        if [ ${RUN_20B} -eq 1 ]; then
            run_step "niah" "20b" "inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_results.csv" "output" \
                python run/run_niah_extension.py \
                    --provider gptoss \
                    --input-path "../../${NIAH_PROMPTS_DIR}/niah_prompts_sequential.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_results.csv" \
                    --input-column prompt \
                    --output-column output \
                    --model-name "${MODEL_20B}" \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_20B} \
                    ${TEST_FLAG}
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            run_step "niah" "120b" "inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_results.csv" "output" \
                python run/run_niah_extension.py \
                    --provider gptoss \
                    --input-path "../../${NIAH_PROMPTS_DIR}/niah_prompts_sequential.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_results.csv" \
                    --input-column prompt \
                    --output-column output \
                    --model-name "${MODEL_120B}" \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_120B} \
                    ${TEST_FLAG}
        fi

        # Evaluate with LLM judge
        if [ ${RUN_20B} -eq 1 ]; then
            run_step "niah" "20b" "evaluation" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_evaluated.csv" "llm_judge_output" \
                python evaluate/evaluate_niah_extension.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_results.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_evaluated.csv" \
                    --model-name "${JUDGE_MODEL}" \
                    --output-column output \
                    --question-column question \
                    --correct-answer-column answer
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            run_step "niah" "120b" "evaluation" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_evaluated.csv" "llm_judge_output" \
                python evaluate/evaluate_niah_extension.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_results.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_evaluated.csv" \
                    --model-name "${JUDGE_MODEL}" \
                    --output-column output \
                    --question-column question \
                    --correct-answer-column answer
        fi

        # Visualize
        if [ ${RUN_20B} -eq 1 ]; then
            run_step "niah" "20b" "visualization" "" "" \
                python evaluate/visualize.py \
                    --csv-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_evaluated.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_niah_heatmap.png" \
                    --title "NIAH Performance - GPT-OSS 20B"
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            run_step "niah" "120b" "visualization" "" "" \
                python evaluate/visualize.py \
                    --csv-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_evaluated.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_niah_heatmap.png" \
                    --title "NIAH Performance - GPT-OSS 120B"
        fi

        cd ../..
        log "NIAH Extension experiment complete!"
    fi

    # ========================================================================
    # EXPERIMENT 2: LongMemEval
    # ========================================================================

    if [ ${RUN_LONGMEMEVAL} -eq 1 ]; then
        log "=========================================="
        log "Starting LongMemEval Experiment"
        log "=========================================="

        cd experiments/longmemeval

        if [ ${RUN_20B} -eq 1 ]; then
            run_step "longmemeval" "20b" "focused_inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_focused_results.csv" "output" \
                python run/run_longmemeval.py \
                    --provider gptoss \
                    --input-path "../../${DATA_DIR}/cleaned_longmemeval_s_focused.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_focused_results.csv" \
                    --input-column focused_prompt \
                    --output-column output \
                    --model-name "${MODEL_20B}" \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_20B} \
                    ${TEST_FLAG}

            run_step "longmemeval" "20b" "full_inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_full_results.csv" "output" \
                python run/run_longmemeval.py \
                    --provider gptoss \
                    --input-path "../../${DATA_DIR}/cleaned_longmemeval_s_full.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_full_results.csv" \
                    --input-column full_prompt \
                    --output-column output \
                    --model-name "${MODEL_20B}" \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_20B} \
                    ${TEST_FLAG} ${TRUNCATE_FLAG}

            run_step "longmemeval" "20b" "focused_evaluation" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_focused_evaluated.csv" "llm_judge_output" \
                python evaluate/evaluate_longmemeval.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_focused_results.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_focused_evaluated.csv" \
                    --model-name "${JUDGE_MODEL}" \
                    --output-column output \
                    --question-column question \
                    --correct-answer-column answer

            run_step "longmemeval" "20b" "full_evaluation" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_full_evaluated.csv" "llm_judge_output" \
                python evaluate/evaluate_longmemeval.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_full_results.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_full_evaluated.csv" \
                    --model-name "${JUDGE_MODEL}" \
                    --output-column output \
                    --question-column question \
                    --correct-answer-column answer

            run_step "longmemeval" "20b" "visualization" "" "" \
                python evaluate/visualize.py \
                    --focused-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_focused_evaluated.csv" \
                    --full-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval_full_evaluated.csv" \
                    --model-name "GPT-OSS 20B" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_longmemeval.png"
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            run_step "longmemeval" "120b" "focused_inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_focused_results.csv" "output" \
                python run/run_longmemeval.py \
                    --provider gptoss \
                    --input-path "../../${DATA_DIR}/cleaned_longmemeval_s_focused.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_focused_results.csv" \
                    --input-column focused_prompt \
                    --output-column output \
                    --model-name "${MODEL_120B}" \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_120B} \
                    ${TEST_FLAG}

            run_step "longmemeval" "120b" "full_inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_full_results.csv" "output" \
                python run/run_longmemeval.py \
                    --provider gptoss \
                    --input-path "../../${DATA_DIR}/cleaned_longmemeval_s_full.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_full_results.csv" \
                    --input-column full_prompt \
                    --output-column output \
                    --model-name "${MODEL_120B}" \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_120B} \
                    ${TEST_FLAG} ${TRUNCATE_FLAG}

            run_step "longmemeval" "120b" "focused_evaluation" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_focused_evaluated.csv" "llm_judge_output" \
                python evaluate/evaluate_longmemeval.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_focused_results.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_focused_evaluated.csv" \
                    --model-name "${JUDGE_MODEL}" \
                    --output-column output \
                    --question-column question \
                    --correct-answer-column answer

            run_step "longmemeval" "120b" "full_evaluation" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_full_evaluated.csv" "llm_judge_output" \
                python evaluate/evaluate_longmemeval.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_full_results.csv" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_full_evaluated.csv" \
                    --model-name "${JUDGE_MODEL}" \
                    --output-column output \
                    --question-column question \
                    --correct-answer-column answer

            run_step "longmemeval" "120b" "visualization" "" "" \
                python evaluate/visualize.py \
                    --focused-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_focused_evaluated.csv" \
                    --full-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval_full_evaluated.csv" \
                    --model-name "GPT-OSS 120B" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_longmemeval.png"
        fi

        cd ../..
        log "LongMemEval experiment complete!"
    fi

    # ========================================================================
    # EXPERIMENT 3: Repeated Words
    # ========================================================================

    if [ ${RUN_REPEATED_WORDS} -eq 1 ]; then
        log "=========================================="
        log "Starting Repeated Words Experiment"
        log "=========================================="
        log "WARNING: This experiment takes significant time due to high output token requirements."

        cd experiments/repeated_words

        if [ ${RUN_20B} -eq 1 ]; then
            run_step "repeated_words" "20b" "inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv" "output" \
                python run/run_repeated_words.py \
                    --provider gptoss \
                    --model-name "${MODEL_20B}" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv" \
                    --common-word "${COMMON_WORD}" \
                    --modified-word "${MODIFIED_WORD}" \
                    --model-max-output-tokens ${MODEL_MAX_OUTPUT_TOKENS} \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_20B} \
                    ${TEST_FLAG}

            run_step "repeated_words" "20b" "evaluation" "" "" \
                python evaluate/evaluate_repeated_words.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv" \
                    --output-dir "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_20b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}_evaluated" \
                    --common-word "${COMMON_WORD}" \
                    --modified-word "${MODIFIED_WORD}" \
                    --model-name "GPT-OSS 20B"
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            run_step "repeated_words" "120b" "inference" \
                "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv" "output" \
                python run/run_repeated_words.py \
                    --provider gptoss \
                    --model-name "${MODEL_120B}" \
                    --output-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv" \
                    --common-word "${COMMON_WORD}" \
                    --modified-word "${MODIFIED_WORD}" \
                    --model-max-output-tokens ${MODEL_MAX_OUTPUT_TOKENS} \
                    --max-context-length ${MAX_CONTEXT_LENGTH} \
                    --max-tokens-per-minute ${RATE_LIMIT_120B} \
                    ${TEST_FLAG}

            run_step "repeated_words" "120b" "evaluation" "" "" \
                python evaluate/evaluate_repeated_words.py \
                    --input-path "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv" \
                    --output-dir "../../${RESULTS_DIR}/${FILE_PREFIX}gpt_oss_120b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}_evaluated" \
                    --common-word "${COMMON_WORD}" \
                    --modified-word "${MODIFIED_WORD}" \
                    --model-name "GPT-OSS 120B"
        fi

        cd ../..
        log "Repeated Words experiment complete!"
    fi

    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================

    log ""
    log "=========================================="
    log "ALL EXPERIMENTS COMPLETE!"
    log "=========================================="
    log ""
    show_workflow_status
    log ""
    log "Results saved to: ${RESULTS_DIR}/"
    log ""
    log "Next steps:"
    log "  1. Review visualizations in ${RESULTS_DIR}/"
    log "  2. Analyze evaluated CSV files for detailed metrics"
    log "  3. Compare results between 20b and 120b models"
    log ""
    log "State file saved to: ${STATE_FILE}"
    log "To clear workflow state: rm ${STATE_FILE}"
    log ""
}

# Run main function
main "$@"
