#! /bin/bash
START_TIME=$SECONDS

# --- Configuration ---
# 1. Path to the root of the Pai-Megatron-Patch directory
MEGATRON_PATCH_PATH=$(pwd)/../..  # Assumes script is run from toolkits/pretrain_data_preprocessing

# 2. Path to the directory containing the input joblib files (e.g., CN-F_data.joblib)
INPUT_DATA_DIR="/ssdshare/share/cs/data/processed_data/val"

# 3. Path to the PRETRAINED KronosTokenizer model directory (containing config.json, pytorch_model.bin)
TOKENIZER_PATH="/ssdshare/share/cs/Kronos/result/ours/S1_9_S2_9_NH128_NL3_NRH256_TAGhf_tot_c/checkpoints/best_model"

# 4. Directory where the output .bin and .idx files will be saved
#    The script will create files like <OUTPUT_DIR>/<market>_<freq>.bin/idx
OUTPUT_DIR="/ssdshare/share/cs/data/Kronos_Data_Mcore/val"

# 5. List of markets to process (space-separated)
# MARKETS_TO_PROCESS="CN-F Crypto-S CN-ETF"
# MARKETS_TO_PROCESS="CN-A US-Eq Crypto-P"
# MARKETS_TO_PROCESS="CN-F" # 最小的数据集
MARKETS_TO_PROCESS="CN-F Crypto-S CN-ETF CN-A US-Eq Crypto-P" # 所有数据集

# 6. Sequence length for chunking
SEQUENCE_LENGTH=512

# 7. Number of worker processes (set to 1 for initial CPU testing)
NUM_WORKERS=4

# --- End Configuration ---


# Set up Python path
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240405 # Adjust if your Megatron-LM path is different
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:${PYTHONPATH}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

echo "Starting Kronos data preprocessing..."
echo "Input data directory: ${INPUT_DATA_DIR}"
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Markets: ${MARKETS_TO_PROCESS}"
echo "Sequence Length: ${SEQUENCE_LENGTH}"
echo "Workers: ${NUM_WORKERS}"

# Run the processing script
python kronos_preprocess_data.py \
  --input ${INPUT_DATA_DIR} \
  --output-prefix ${OUTPUT_DIR} \
  --tokenizer-path ${TOKENIZER_PATH} \
  --markets ${MARKETS_TO_PROCESS} \
  --sequence-length ${SEQUENCE_LENGTH} \
  --dataset-impl mmap \
  --workers ${NUM_WORKERS} \
  --gpu-batch-size 64

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "--------------------------------------------------"
echo "Preprocessing finished!"
echo "Total time: $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
echo "Output files are in: ${OUTPUT_DIR}"
echo "--------------------------------------------------"
