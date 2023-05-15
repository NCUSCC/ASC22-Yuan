#! /bin/bash

# Change for multinode config
MP_SIZE=1

DEBUG=1
if [[ ${DEBUG} == 1 ]];  then
       MP_SIZE=4
       NUM_WORKERS=1
       NUM_GPUS_PER_WORKER=4
       HIDDEN_SIZE=3072
       NUM_ATTN_HEADS=24
       NUM_LAYERS=40
       BATCHSIZE=1
else
       NUM_WORKERS=${DLTS_NUM_WORKER}
       NUM_GPUS_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER}
       HIDDEN_SIZE=8192
       NUM_ATTN_HEADS=32
       NUM_LAYERS=50
       BATCHSIZE=4

       #HIDDEN_SIZE=4096
       #NUM_LAYERS=24 # 50
       #BATCHSIZE=16
fi


BASE_DATA_PATH=/data/Megatron-LM/data
DATA_PATH=/home/asc20g0/ASC22/my-gpt2/my-gpt2_text_sentence
VOCAB_PATH=/home/asc20g0/ASC22/vocab_dxy.txt
#MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_4.7B_ds-zero3

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
if [[ -z $1 ]]; then
       config_json="$script_dir/ds_zero_stage_3_config.json"

       # offloads to NVMe
       #config_json="$script_dir/ds_zero_stage_infinity_config.json"
else
       config_json=$script_dir/`basename $1`
fi

#ZeRO Configs
stage=3
reduce_scatter=true
contigious_gradients=true
rbs=500000000
agbs=500000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=true
CC=true
SYNCHRONIZE=true
PROFILE=false

# TiledLinear splits, 0 is disable
TILED_LINEAR="false"
TILE_DIM=1


# Megatron Model Parallelism
LOGDIR="tboard-zero3/stage${stage}-lazyscatter-${NUM_LAYERS}l_${HIDDEN_SIZE}h_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${BATCHSIZE}b"


gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --batch-size $BATCHSIZE \
        --train-iters 320000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file  "/home/asc20g0/ASC22/vocab_dxy.txt" \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 2000 \
        --eval-iters 10 \
        --fp16 \
        --scattered-embeddings \
        --split-transformers \
	--train-tokens 1000000 \
        --tokenizer-type 'BertWordPieceLowerCase'
"
        #--tensorboard-dir ${LOGDIR}

 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs}
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

if [ "${TILED_LINEAR}" = "true" ]; then
tile_opt="${tile_opt} \
        --memory-centric-tiled-linear \
        --tile-factor=${TILE_DIM}"
fi


full_options="${gpt_options} ${deepspeed_options} ${chkp_opt} ${tile_opt}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
