#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config

MASTER_ADDR=localhost
NNODES=1
#MASTER_ADDR=192.168.1.72
MASTER_PORT=6005
#NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/asc22g0/ASC22/my-gpt2/my-gpt2_text_sentence
CHECKPOINT_PATH=/home/asc22g0/ASC22/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/checkpoints/ds_gpt2_4.7B

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --model-parallel-size 4 \
       --num-layers 40 \
       --hidden-size 3072 \
       --num-attention-heads 24\
       --batch-size 2 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /home/asc22g0/ASC22/vocab_dxy.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --train-tokens 5000000 \
       --tokenizer-type BertWordPieceLowerCase 




set +x
