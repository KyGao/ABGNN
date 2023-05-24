if [ -z ${CDR} ]; then CDR='1'; fi

SABDAB_DATA_DIR=dataset/exp1/cdrh${CDR}
PRETRAINED_FILE=checkpoints/pretrained/checkpoint215.pt
SAVE_DIR=checkpoints/exp1-cdrh${CDR}

FAIRSEQ_MODELS_DIR=fairseq_models

PEAK_LR=0.0001
MAX_EPOCH=40
PATIENCE=40

MAX_SENTENCES=1
UPDATE_FREQ=1
MAX_POSITIONS=256

PREFIX_LEN=5
ITER_NUM=5

LOSS_ENC_S=1
LOSS_DEC_S=1
LOSS_DEC_X=1

SEED=42

# training
echo $(which fairseq-train) 
fairseq-train --sabdab-data $SABDAB_DATA_DIR \
  --user-dir $FAIRSEQ_MODELS_DIR --finetune \
  --cdr-type ${CDR} \
  --task antibody_generation_task \
  --criterion antibody_generation_loss \
  --arch antibody_roberta_base \
  --finetune-bert-scheme prefix_tuning --pre-seq-len $PREFIX_LEN \
  --refine-iteration $ITER_NUM --block_size 8 \
  --loss-scale-enc $LOSS_ENC_S --loss-scale-dec-sloss $LOSS_DEC_S --loss-scale-dec-xloss $LOSS_DEC_X \
  --optimizer adam --clip-norm 1.0 \
  --lr-scheduler fixed --lr $PEAK_LR --force-anneal 1 --lr-shrink 0.95 \
  --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ --max-positions $MAX_POSITIONS --max-epoch $MAX_EPOCH \
  --log-format simple --log-interval 5 \
  --valid-subset valid,test --skip-invalid-size-inputs-valid-test \
  --save-interval 1 --save-dir $SAVE_DIR \
  --finetune-from-model $PRETRAINED_FILE \
  --patience $PATIENCE --tensorboard-logdir $SAVE_DIR/tensorboard  \
  --seed $SEED --num-workers 0


# inference
python inference.py --cdr_type ${CDR} \
    --cktpath $SAVE_DIR/checkpoint_best.pt --data_path dataset/exp1/cdrh${CDR} > $SAVE_DIR/test_best.txt
for infer_epoch in {1..40}
do
    python inference.py --cdr_type ${CDR} \
        --cktpath $SAVE_DIR/checkpoint${infer_epoch}.pt --data_path dataset/exp1/cdrh${CDR} > $SAVE_DIR/test${infer_epoch}.txt
done

