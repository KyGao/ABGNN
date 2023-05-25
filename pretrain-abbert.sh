SEQ_DATA_DIR=data/fairseq-oas/seq
TAG_DATA_DIR=data/fairseq-oas/tag
FAIRSEQ_MODELS_DIR=fairseq_models

TOTAL_UPDATES=2000000    # Total number of training steps
WARMUP_UPDATES=${1:-10000}    # Warmup the learning rate over this many updates
PEAK_LR=${2:-0.0006}         # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=256   # Max sequence length
MAX_POSITIONS=256       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=${3:-64}        # Number of sequences per batch (batch size)
UPDATE_FREQ=${4:-16}         # Increase the batch size 16x
SAMPLE_MODE=${5:-"none"}
CLIP_NORM=${6:-0.0}
# The final real batch size is MAX_SENTENCES x GPU_NUM x UPDATE_FREQ

MODEL_NAME=abbert_warmup${WARMUP_UPDATES}_lr${PEAK_LR}_maxsen${MAX_SENTENCES}_upfreq${UPDATE_FREQ}_sample${SAMPLE_MODE}_clip${CLIP_NORM}
MODLE_PATH=checkpoints/pretrained/$MODEL_NAME

export MKL_THREADING_LAYER=GNU
echo $(which fairseq-train) 

python \
    $(which fairseq-train) $SEQ_DATA_DIR \
    --tag-data $TAG_DATA_DIR \
    --user-dir $FAIRSEQ_MODELS_DIR \
    --task antibody_masked_lm \
    --criterion antibody_masked_lm \
    --arch antibody_roberta_base \
    --sample-break-mode $SAMPLE_MODE \
    --mask-prob 0.5 \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --adam-eps 1e-6 \
    --clip-norm ${CLIP_NORM} \
    --lr-scheduler polynomial_decay \
    --lr $PEAK_LR \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES \
    --log-format simple \
    --log-interval 1 \
    --skip-invalid-size-inputs-valid-test\
    --keep-last-epochs 100\
    --save-dir $MODLE_PATH \
    --tensorboard-logdir $MODLE_PATH/tensorboard
