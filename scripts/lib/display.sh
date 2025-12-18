#!/bin/bash
# Display functions for Context Rot Research Workflow
# Created: 2025-11-28

# ============================================================================
# BANNERS & NOTIFICATIONS
# ============================================================================

show_test_mode_banner() {
    if [ ${TEST_MODE} -eq 1 ]; then
        log ""
        log "=========================================="
        log "TEST MODE ENABLED"
        log "=========================================="
        log "Running validation with reduced samples"
        log "Results will be prefixed with 'test_'"
        log "Expected completion: 10-20 minutes"
        log ""
        log "Sample counts (test vs production):"
        log "  NIAH Extension:    12 samples (vs 88)"
        log "  LongMemEval:       40 samples (vs 612)"
        log "  Repeated Words:    ~15 samples (vs ~300)"
        log ""
        log "To run full research:"
        log "  ./scripts/run_full_research.sh"
        log "=========================================="
        log ""
    fi
}

show_token_tracking_banner() {
    local prefix=""
    if [ ${TEST_MODE} -eq 1 ]; then
        prefix="test_"
    fi

    log "=========================================="
    log "Token Tracking Enabled"
    log "=========================================="
    log "Real-time token usage tracking is active for all experiments."
    log ""
    log "During execution, live dashboards will be created:"
    log "  results/${prefix}*_token_dashboard.txt"
    log ""
    log "Monitor live token usage in another terminal:"
    log "  tail -f results/${prefix}gpt_oss_20b_niah_results_token_dashboard.txt"
    log ""
    log "Token summaries (JSON) will be saved after each experiment:"
    log "  results/${prefix}*_token_summary.json"
    log "=========================================="
    log ""
}

# ============================================================================
# COST ESTIMATION
# ============================================================================

show_cost_estimate() {
    if [ ${TEST_MODE} -eq 1 ]; then
        log "=========================================="
        log "TEST MODE - Cost & Time Estimation"
        log "=========================================="
        log ""
        log "Estimated costs:"
        log "  Inference: \$0.20 - \$1.00 (depending on deployment)"
        log "  LLM Judge: \$0.10 - \$0.50"
        log "  TOTAL:     \$0.50 - \$2.00"
        log ""
        log "Estimated time: 10-20 minutes"
        log "=========================================="
        log ""
        return
    fi

    log "=========================================="
    log "Cost Estimation"
    log "=========================================="

    local total_inference_cost_low=0
    local total_inference_cost_high=0
    local total_judge_cost=5

    if [ ${RUN_20B} -eq 1 ]; then
        if [ "$DEPLOY_20B" == "local" ]; then
            log "20b model (local): Hardware costs only (electricity, compute depreciation)"
        else
            log "20b model (cloud): Estimated \$20-50 (varies by provider and token usage)"
            total_inference_cost_low=$((total_inference_cost_low + 20))
            total_inference_cost_high=$((total_inference_cost_high + 50))
        fi
    fi

    if [ ${RUN_120B} -eq 1 ]; then
        if [ "$DEPLOY_120B" == "local" ]; then
            log "120b model (local): Hardware costs only (requires high-end GPU)"
        else
            log "120b model (cloud): Estimated \$50-150 (varies by provider and token usage)"
            total_inference_cost_low=$((total_inference_cost_low + 50))
            total_inference_cost_high=$((total_inference_cost_high + 150))
        fi
    fi

    log ""
    log "LLM Judge (${JUDGE_MODEL}): ~\$${total_judge_cost}-15"
    log ""

    if [ $total_inference_cost_high -gt 0 ]; then
        log "TOTAL ESTIMATED COST: \$${total_inference_cost_low}-\$${total_inference_cost_high} (cloud inference) + \$${total_judge_cost}-15 (judge)"
    else
        log "TOTAL ESTIMATED COST: Hardware costs only (local) + \$${total_judge_cost}-15 (judge)"
    fi

    log ""
    log "Notes:"
    log "  - Local deployment: Free API costs, but requires hardware (GPU/CPU)"
    log "  - Cloud deployment: Pay per token, varies by provider rate limits"
    log "  - LongMemEval has highest token usage (~34.6M input tokens per model)"
    log "  - Repeated Words has high output token usage (~32k tokens per test)"
    log "  - Judge costs are for OpenAI gpt-4o-mini (~\$0.15/1M tokens)"
    log "=========================================="
    log ""
}

# ============================================================================
# TOKEN TRACKING
# ============================================================================

start_dashboard_viewer() {
    local dashboard_file=$1
    local experiment_name=$2

    log "Live token tracking enabled for ${experiment_name}"
    log "Dashboard file: ${dashboard_file}"
    log ""
    log "To monitor live token usage, run in another terminal:"
    log "  tail -f ${dashboard_file}"
    log ""
}

show_final_token_summary() {
    local summary_file=$1

    if [ ! -f "$summary_file" ]; then
        return
    fi

    log ""
    log "=========================================="
    log "Token Usage Summary"
    log "=========================================="

    local total_calls=$(grep '"total_calls"' "$summary_file" | sed 's/.*: //' | tr -d ',')
    local total_tokens=$(grep '"total_tokens"' "$summary_file" | sed 's/.*: //' | tr -d ',')
    local total_cost=$(grep '"total_cost_usd"' "$summary_file" | sed 's/.*: //' | tr -d ',')
    local elapsed_min=$(grep '"elapsed_minutes"' "$summary_file" | sed 's/.*: //' | tr -d ',')

    log "API calls:      ${total_calls}"
    log "Total tokens:   ${total_tokens}"
    log "Cost:           \$${total_cost}"
    log "Duration:       ${elapsed_min} minutes"
    log ""
    log "Full summary:   ${summary_file}"
    log "=========================================="
    log ""
}

# ============================================================================
# WORKFLOW STATUS DASHBOARD
# ============================================================================

show_workflow_status() {
    log "=========================================="
    log "Workflow Status Dashboard"
    log "=========================================="

    # NIAH Extension
    if [ ${RUN_NIAH} -eq 1 ]; then
        log ""
        log "NIAH Extension:"

        if [ ${RUN_20B} -eq 1 ]; then
            local niah_20b_inf="${RESULTS_DIR}/gpt_oss_20b_niah_results.csv"
            local niah_20b_eval="${RESULTS_DIR}/gpt_oss_20b_niah_evaluated.csv"
            local niah_20b_viz="${RESULTS_DIR}/gpt_oss_20b_niah_heatmap.png"

            if is_csv_complete "$niah_20b_eval" "llm_judge_output" && [ -f "$niah_20b_viz" ]; then
                log "  [done] 20b model - complete ($(count_csv_rows "$niah_20b_eval" "llm_judge_output"))"
            elif is_csv_complete "$niah_20b_inf" "output"; then
                log "  [....] 20b model - inference done ($(count_csv_rows "$niah_20b_inf" "output")), evaluation pending"
            elif [ -f "$niah_20b_inf" ]; then
                log "  [....] 20b model - in progress ($(count_csv_rows "$niah_20b_inf" "output"))"
            else
                log "  [    ] 20b model - not started"
            fi
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            local niah_120b_inf="${RESULTS_DIR}/gpt_oss_120b_niah_results.csv"
            local niah_120b_eval="${RESULTS_DIR}/gpt_oss_120b_niah_evaluated.csv"
            local niah_120b_viz="${RESULTS_DIR}/gpt_oss_120b_niah_heatmap.png"

            if is_csv_complete "$niah_120b_eval" "llm_judge_output" && [ -f "$niah_120b_viz" ]; then
                log "  [done] 120b model - complete ($(count_csv_rows "$niah_120b_eval" "llm_judge_output"))"
            elif is_csv_complete "$niah_120b_inf" "output"; then
                log "  [....] 120b model - inference done ($(count_csv_rows "$niah_120b_inf" "output")), evaluation pending"
            elif [ -f "$niah_120b_inf" ]; then
                log "  [....] 120b model - in progress ($(count_csv_rows "$niah_120b_inf" "output"))"
            else
                log "  [    ] 120b model - not started"
            fi
        fi
    fi

    # LongMemEval
    if [ ${RUN_LONGMEMEVAL} -eq 1 ]; then
        log ""
        log "LongMemEval:"

        if [ ${RUN_20B} -eq 1 ]; then
            local longmem_20b_foc_eval="${RESULTS_DIR}/gpt_oss_20b_longmemeval_focused_evaluated.csv"
            local longmem_20b_full_eval="${RESULTS_DIR}/gpt_oss_20b_longmemeval_full_evaluated.csv"
            local longmem_20b_viz="${RESULTS_DIR}/gpt_oss_20b_longmemeval.png"

            if is_csv_complete "$longmem_20b_foc_eval" "llm_judge_output" && \
               is_csv_complete "$longmem_20b_full_eval" "llm_judge_output" && \
               [ -f "$longmem_20b_viz" ]; then
                log "  [done] 20b model - complete"
            else
                local foc_count=$(count_csv_rows "${RESULTS_DIR}/gpt_oss_20b_longmemeval_focused_results.csv" "output")
                local full_count=$(count_csv_rows "${RESULTS_DIR}/gpt_oss_20b_longmemeval_full_results.csv" "output")
                log "  [....] 20b model - focused: $foc_count, full: $full_count"
            fi
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            local longmem_120b_foc_eval="${RESULTS_DIR}/gpt_oss_120b_longmemeval_focused_evaluated.csv"
            local longmem_120b_full_eval="${RESULTS_DIR}/gpt_oss_120b_longmemeval_full_evaluated.csv"
            local longmem_120b_viz="${RESULTS_DIR}/gpt_oss_120b_longmemeval.png"

            if is_csv_complete "$longmem_120b_foc_eval" "llm_judge_output" && \
               is_csv_complete "$longmem_120b_full_eval" "llm_judge_output" && \
               [ -f "$longmem_120b_viz" ]; then
                log "  [done] 120b model - complete"
            else
                local foc_count=$(count_csv_rows "${RESULTS_DIR}/gpt_oss_120b_longmemeval_focused_results.csv" "output")
                local full_count=$(count_csv_rows "${RESULTS_DIR}/gpt_oss_120b_longmemeval_full_results.csv" "output")
                log "  [....] 120b model - focused: $foc_count, full: $full_count"
            fi
        fi
    fi

    # Repeated Words
    if [ ${RUN_REPEATED_WORDS} -eq 1 ]; then
        log ""
        log "Repeated Words:"

        if [ ${RUN_20B} -eq 1 ]; then
            local rw_20b="${RESULTS_DIR}/gpt_oss_20b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv"
            local rw_20b_eval="${RESULTS_DIR}/gpt_oss_20b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}_evaluated"

            if [ -d "$rw_20b_eval" ]; then
                log "  [done] 20b model - complete"
            elif [ -f "$rw_20b" ]; then
                log "  [....] 20b model - inference ($(count_csv_rows "$rw_20b" "output"))"
            else
                log "  [    ] 20b model - not started"
            fi
        fi

        if [ ${RUN_120B} -eq 1 ]; then
            local rw_120b="${RESULTS_DIR}/gpt_oss_120b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}.csv"
            local rw_120b_eval="${RESULTS_DIR}/gpt_oss_120b_repeated_words_${COMMON_WORD}_${MODIFIED_WORD}_evaluated"

            if [ -d "$rw_120b_eval" ]; then
                log "  [done] 120b model - complete"
            elif [ -f "$rw_120b" ]; then
                log "  [....] 120b model - inference ($(count_csv_rows "$rw_120b" "output"))"
            else
                log "  [    ] 120b model - not started"
            fi
        fi
    fi

    log ""
    log "=========================================="
}
