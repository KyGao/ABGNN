python covid-optimize.py \
    --cktpath checkpoints/exp3-sabdab/checkpoint5.pt \
    --save_dir checkpoints/exp3-ckpts/exp3-sabdab-bsz64-checkpoint5.pt \
    --cluster data/finetune/exp3-covabdab/cdrh3_split.txt \
    --topk 64 --epochs 20000