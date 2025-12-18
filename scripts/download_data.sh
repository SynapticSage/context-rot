#!/bin/bash
# Download required datasets for Context Rot Research
# Created: 2025-11-28
#
# Usage:
#   ./scripts/download_data.sh        # Download all datasets
#   ./scripts/download_data.sh niah   # Download only NIAH dataset
#   ./scripts/download_data.sh longmemeval  # Download only LongMemEval dataset
#
# If automatic download fails, manual download links are provided.

set -u  # Exit on undefined variable (but not on command failure)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DATA_DIR="./data"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check for gdown (Google Drive downloader)
check_gdown() {
    if ! command -v gdown &> /dev/null; then
        log "gdown not found. Installing..."
        pip install gdown
    fi
}

# Download PaulGrahamEssays for NIAH experiment
download_niah() {
    local target_dir="${DATA_DIR}/PaulGrahamEssays"
    local folder_id="14uHYF65yu7cNGANungZX1NRboqwHHuVB"
    local folder_url="https://drive.google.com/drive/folders/${folder_id}"

    if [ -d "$target_dir" ] && [ "$(ls -1 $target_dir/*.txt 2>/dev/null | wc -l)" -gt 0 ]; then
        local count=$(ls -1 $target_dir/*.txt 2>/dev/null | wc -l)
        log "PaulGrahamEssays already exists at $target_dir ($count txt files)"
        log "  To re-download, remove the directory first: rm -rf $target_dir"
        return 0
    fi

    log "Downloading PaulGrahamEssays dataset..."
    log "Source: $folder_url"

    mkdir -p "$target_dir"

    # Try gdown - download to parent and let it create the folder
    if gdown --folder "$folder_url" -O "${DATA_DIR}" --remaining-ok 2>/dev/null; then
        # Fix nested folder structure if gdown created it
        if [ -d "$target_dir/PaulGrahamEssays" ]; then
            mv "$target_dir/PaulGrahamEssays"/* "$target_dir/" 2>/dev/null || true
            rmdir "$target_dir/PaulGrahamEssays" 2>/dev/null || true
        fi

        local file_count=$(ls -1 "$target_dir"/*.txt 2>/dev/null | wc -l)
        if [ "$file_count" -gt 0 ]; then
            log "Successfully downloaded $file_count files to $target_dir"
            return 0
        fi
    fi

    # If gdown fails, provide manual instructions
    log ""
    log "=========================================="
    log "AUTOMATIC DOWNLOAD FAILED"
    log "=========================================="
    log ""
    log "Please download manually:"
    log "  1. Open: $folder_url"
    log "  2. Download all files"
    log "  3. Extract to: $target_dir"
    log ""
    log "Alternative: Use the original research data source"
    log "  https://huggingface.co/datasets/paulgraham/essays"
    log ""
    return 1
}

# Download LongMemEval datasets (primary: Hugging Face, fallback: Google Drive)
download_longmemeval() {
    local focused_file="${DATA_DIR}/cleaned_longmemeval_s_focused.csv"
    local full_file="${DATA_DIR}/cleaned_longmemeval_s_full.csv"
    local hf_dataset="kellyhongg/cleaned-longmemeval-s"
    local folder_url="https://drive.google.com/drive/folders/1AS1oytdcCH3y6p-DNuaNYI7I48IFgbfe"

    if [ -f "$focused_file" ] && [ -f "$full_file" ]; then
        log "LongMemEval datasets already exist"
        log "  To re-download, remove files first:"
        log "    rm $focused_file $full_file"
        return 0
    fi

    log "Downloading LongMemEval datasets..."
    log "Primary source: Hugging Face ($hf_dataset)"

    mkdir -p "$DATA_DIR"

    # Try Hugging Face first (more reliable than Google Drive)
    if python3 -c "
from datasets import load_dataset
import pandas as pd

ds = load_dataset('$hf_dataset')
df = ds['train'].to_pandas()

# Map HF columns to expected format
focused_df = pd.DataFrame({
    'custom_id': df['custom_id'],
    'focused_prompt': df['focused_input'],
    'token_count': df['focused_input_tokens'],
    'question': df['question'],
    'answer': df['answer']
})
focused_df.to_csv('$focused_file', index=False)

full_df = pd.DataFrame({
    'custom_id': df['custom_id'],
    'full_prompt': df['full_input'],
    'token_count': df['full_input_tokens'],
    'question': df['question'],
    'answer': df['answer']
})
full_df.to_csv('$full_file', index=False)
print(f'Downloaded {len(df)} samples from Hugging Face')
" 2>/dev/null; then
        if [ -f "$focused_file" ] && [ -f "$full_file" ]; then
            log "Downloaded from Hugging Face: cleaned_longmemeval_s_focused.csv"
            log "Downloaded from Hugging Face: cleaned_longmemeval_s_full.csv"
            return 0
        fi
    fi

    log "Hugging Face download failed, trying Google Drive..."

    # Fallback to gdown
    if gdown --folder "$folder_url" -O "$DATA_DIR" --remaining-ok 2>/dev/null; then
        if [ -f "$focused_file" ] && [ -f "$full_file" ]; then
            log "Downloaded from Google Drive"
            return 0
        fi
    fi

    # If all fail, provide manual instructions
    log ""
    log "=========================================="
    log "AUTOMATIC DOWNLOAD FAILED"
    log "=========================================="
    log ""
    log "Please download manually from one of these sources:"
    log ""
    log "Option 1 - Hugging Face (recommended):"
    log "  pip install datasets"
    log "  python -c \"from datasets import load_dataset; ds = load_dataset('$hf_dataset')\""
    log ""
    log "Option 2 - Google Drive:"
    log "  1. Open: $folder_url"
    log "  2. Download: cleaned_longmemeval_s_focused.csv, cleaned_longmemeval_s_full.csv"
    log "  3. Place them in: $DATA_DIR/"
    log ""
    return 1
}

# Show usage
usage() {
    echo "Usage: $0 [dataset]"
    echo ""
    echo "Datasets:"
    echo "  all         Download all datasets (default)"
    echo "  niah        Download PaulGrahamEssays for NIAH experiment"
    echo "  longmemeval Download LongMemEval datasets"
    echo ""
    echo "Examples:"
    echo "  $0              # Download everything"
    echo "  $0 niah         # Download only NIAH dataset"
    echo "  $0 longmemeval  # Download only LongMemEval dataset"
    exit 0
}

# Main
main() {
    local dataset="${1:-all}"

    case "$dataset" in
        -h|--help|help)
            usage
            ;;
        all)
            log "=========================================="
            log "Downloading all required datasets"
            log "=========================================="
            check_gdown
            mkdir -p "$DATA_DIR"

            local niah_ok=0
            local longmemeval_ok=0

            download_niah && niah_ok=1
            download_longmemeval && longmemeval_ok=1

            log ""
            log "=========================================="
            log "Download Summary"
            log "=========================================="
            if [ $niah_ok -eq 1 ]; then
                log "  NIAH (PaulGrahamEssays): OK"
            else
                log "  NIAH (PaulGrahamEssays): MANUAL DOWNLOAD REQUIRED"
            fi
            if [ $longmemeval_ok -eq 1 ]; then
                log "  LongMemEval: OK"
            else
                log "  LongMemEval: MANUAL DOWNLOAD REQUIRED"
            fi
            log ""
            log "Data directory contents:"
            ls -la "$DATA_DIR" 2>/dev/null || log "  (directory empty or not found)"
            ;;
        niah)
            log "Downloading NIAH dataset only..."
            check_gdown
            mkdir -p "$DATA_DIR"
            download_niah
            ;;
        longmemeval)
            log "Downloading LongMemEval dataset only..."
            check_gdown
            mkdir -p "$DATA_DIR"
            download_longmemeval
            ;;
        *)
            log "ERROR: Unknown dataset: $dataset"
            usage
            ;;
    esac
}

main "$@"
