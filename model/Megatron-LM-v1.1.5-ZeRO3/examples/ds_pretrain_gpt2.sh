#! /bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

DATA_PATH=/home/asc22g0/ASC22/my-gpt2/my-gpt2_text_sentence
CHECKPOINT_PATH=/home/asc22g0/ASC22/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/checkpoints/ds_gpt2_4.7B-zero

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_zero_stage_1.json"

# Megatron Model Parallelism
mp_size=4

NLAYERS=40
NHIDDEN=3072
BATCHSIZE=2
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4"

#ZeRO Configs
stage=2
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false


gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads 24 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --batch-size 2 \
        --train-iters 320000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file /home/asc22g0/ASC22/vocab_dxy.txt \
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
        --log-interval 100 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --fp16 \
	--train-tokens 1000000 \
	--tokenizer-type BertWordPieceLowerCase
	"
        --tensorboard-dir ${LOGDIR}

  
 deepspeed_options=" \
                --deepspeed \
		--deepspeed_config ${config_json} \		
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "
#--deepspeed_config ${config_json} \
if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
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

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
