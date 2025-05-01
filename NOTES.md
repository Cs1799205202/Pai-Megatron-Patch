DATASET_PATH=/ssdshare/share/cs/data/Kronos_Data_Megatron/CN-F_1

DATASET_PATH=/ssdshare/share/cs/data/Kronos_Data_Mcore/CN-F_1

INPUT_PATH=/ssdshare/share/cs/data/processed_data/train
OUTPUT_PATH=/ssdshare/share/cs/data/Kronos_Data_Megatron

TOKEN_MODEL_PATH=/ssdshare/share/cs/Kronos/result/ours/S1_9_S2_9_NH128_NL3_NRH256_TAGhf_tot_c/checkpoints/best_model

1. Llama3.1 要关掉 ROPE scaling，默认传一个空的 dict() 会使用 Llama3.1 的默认 scaling config，需要传 None 关掉
2. 不能开 SP，因为开了后会先转置 (b, s, h) -> (s, b, h) 然后沿 seq_len 维度切分(s/TP, b, h)，而 timestamp 没有切分，是完整的(b, s, h)，维度不匹配，加上广播一通乱七八糟的操作后会烂掉