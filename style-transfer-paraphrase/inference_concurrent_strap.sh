###############
MODEL=WNC_large
TOP_P=0.6
###############

MODEL_DIR=style_paraphrase/models/$MODEL
INPUT=/home/ppommer/repos/master-thesis/neutralizing-bias/src/inference/concurrent/output_concurrent.txt
OUTPUT=inference/concurrent_strap/output_concurrent_strap.txt

python strap_many.py \
    --input $INPUT \
    --output $OUTPUT \
    --model_dir $MODEL_DIR \
    --batch_size 1 \
    --top_p_value $TOP_P

python utils/evaluate.py \
    --pred_data $OUTPUT \
    --output inference/concurrent_strap/stats.txt
