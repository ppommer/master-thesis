###############
MODEL=WNC_large
TOP_P=0.6
###############

MODEL_DIR=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/style_paraphrase/models/$MODEL
INPUT=/home/ppommer/repos/master-thesis/neutralizing-bias/src/inference/concurrent/output_concurrent.txt
OUTPUT=inference/output_concurrent_strap.txt

python strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value $TOP_P \
    --input $INPUT \
    --output $OUTPUT
