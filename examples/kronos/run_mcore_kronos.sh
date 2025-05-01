#!/bin/bash
set -e
ENV=ailab
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
export NCCL_P2P_DISABLE=1
# export UB_SKIPMC=1

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ $ENV = dsw ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=8
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
elif [ $ENV = ailab ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=4
fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
EXTRA_VOCAB_SIZE=0

### BASE CONFIG ###
MODEL_SIZE=100M
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=512
PAD_LEN=512
PR=bf16
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=2
PP=2
CP=1
SP=false
DO=true
FL=true
SFT=false
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=false
OPTIMIZER_OFFLOAD=false
SAVE_INTERVAL=50
# DATASET_PATH=/ssdshare/share/cs/llama3-datasets/wudao_llama3bpe_content_document
# VALID_DATASET_PATH=/ssdshare/share/cs/llama3-datasets/wudao_llama3bpe_content_document
# TRAIN_DATASET_PATH=" \
#     0.7 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_1 \
#     0.3 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_5 \
# "
# VALID_DATASET_PATH=" \
#     0.7 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_1 \
#     0.3 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_5 \
# "
# dt_sample_prob = {
#     'CN-F': {
#         1: 0.1 * 0.8,
#         5: 0.1 * 0.16,
#         15: 0.1 * 0.037,
#         1440: 0.1 * 0.003
#     },
#     'CN-ETF': {
#         1: 0.1 * 0.8,
#         5: 0.1 * 0.195,
#         1440: 0.1 * 0.005
#     },
#     'Crypto-P': {
#         1: 0.20 * 0.65,
#         5: 0.20 * 0.20,
#         15: 0.20 * 0.10,
#         60: 0.20 * 0.045,
#         1440: 0.20 * 0.005
#     },
#     'Crypto-S': {
#         1: 0.05 * 0.65,
#         5: 0.05 * 0.20,
#         15: 0.05 * 0.10,
#         60: 0.05 * 0.045,
#         1440: 0.05 * 0.005
#     },
#     'CN-A': {
#         1: 0.25 * 0.80,
#         5: 0.25 * 0.18,
#         1440: 0.25 * 0.02,
#     },
#     'US-Eq': {
#         1: 0.25 * 0.80,
#         5: 0.25 * 0.18,
#         1440: 0.25 * 0.02
#     }
# }

TRAIN_DATASET_PATH=" \
    0.08 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_1 \
    0.016 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_5 \
    0.0037 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_15 \
    0.0003 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_1440 \
    0.08 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-ETF_1 \
    0.0195 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-ETF_5 \
    0.0005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-ETF_1440 \
    0.13 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_1 \
    0.04 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_5 \
    0.02 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_15 \
    0.009 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_60 \
    0.001 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_1440 \
    0.0325 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_1 \
    0.01 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_5 \
    0.005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_15 \
    0.00225 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_60 \
    0.00025 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_1440 \
    0.2 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-A_1 \
    0.045 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-A_5 \
    0.005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-A_1440 \
    0.2 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/US-Eq_1 \
    0.045 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/US-Eq_5 \
    0.005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/US-Eq_1440 \
"
VALID_DATASET_PATH=" \
    0.08 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_1 \
    0.016 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_5 \
    0.0037 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_15 \
    0.0003 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-F_1440 \
    0.08 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-ETF_1 \
    0.0195 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-ETF_5 \
    0.0005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-ETF_1440 \
    0.13 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_1 \
    0.04 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_5 \
    0.02 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_15 \
    0.009 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_60 \
    0.001 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-P_1440 \
    0.0325 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_1 \
    0.01 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_5 \
    0.005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_15 \
    0.00225 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_60 \
    0.00025 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/Crypto-S_1440 \
    0.2 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-A_1 \
    0.045 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-A_5 \
    0.005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/CN-A_1440 \
    0.2 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/US-Eq_1 \
    0.045 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/US-Eq_5 \
    0.005 /ssdshare/share/cs/data/Kronos_Data_Mcore/train/US-Eq_1440 \
"
# PRETRAIN_CHECKPOINT_PATH=/ssdshare/share/cs/llama3-ckpts/Meta-Llama-3.1-8B/mcore-tp2-pp2
PRETRAIN_CHECKPOINT_PATH=none

# the following two values will not be used when SFT is true
# TRAIN_TOKENS=10000
# WARMUP_TOKENS=100

# 100 * 8 * 512
TRAIN_TOKENS=4096000
WARMUP_TOKENS=0
###############################

OUTPUT_BASEPATH=/ssdshare/share/cs/llama3-ckpts/Meta-Llama-3.1-8B/mcore-tp2-pp2
OUTPUT_BASEPATH=/ssdshare/share/cs/Kronos_Checkpoint/mcore-tp2-pp2
### OTHERS ###

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
fi

if [ $MODEL_SIZE = 20M ]; then

NUM_LAYERS=6
HIDDEN_SIZE=512
NUM_ATTN_HEADS=8
INTERMEDIATE_SIZE=1280
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


elif [ $MODEL_SIZE = 100M ]; then

NUM_LAYERS=12
HIDDEN_SIZE=832
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=2048
NUM_KEY_VALUE_HEADS=16
MAX_POSITION_EMBEDDINGS=131072

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


elif [ $MODEL_SIZE = 8B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


elif [ $MODEL_SIZE = 70B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=28672
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

fi

# TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
TP_COMM_OVERLAP=0

comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

te_options=" \
        --transformer-impl local"

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

if [ $SFT = true ]; then
    TRAIN_ITERS=${23}
    LR_WARMUP_ITERS=${24}
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
    PREFIX="finetune-mcore-llama3-1-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" \
         --eod-mask-loss \
         --calculate-per-token-loss \
         --train-mode finetune"
else
    TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    PREFIX="pretrain-mcore-llama3-1-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" \
        --train-mode pretrain"
fi

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_option=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT"
else 
    # dataset_option=" \
    #     --data-path ${DATASET_PATH} \
    #     --split 99,1,0 \
    #     --dataset MMAP"
    dataset_option=" \
        --train-data-path ${TRAIN_DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset MMAP"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
        --reset-position-ids \
        --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi


##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
# find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}


megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type KronosTokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon 1e-05 \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base 10000 \
        --no-save-optim \
        --wandb-project TimeSeries \
        --wandb-exp-name TimeSeries-mcore-tp2-pp2 \
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_kronos.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${gqa_options} ${offload_option} ${comm_overlap_option} ${sft_option} ${vp_options} ${packing_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
