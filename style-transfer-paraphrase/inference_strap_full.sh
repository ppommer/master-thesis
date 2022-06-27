MODEL_DIR=style_paraphrase/models/WNC_biased_full
INPUT=datasets/WNC/test_biased.txt
OUTPUT_DIR=inference/strap_full

# top_p 0.0
python strap_many.py \
    --input $INPUT \
    --output $OUTPUT_DIR/output_strap_full_0.txt \
    --model_dir $MODEL_DIR \
    --batch_size 1 \
    --top_p_value 0.0

python utils/evaluate.py \
    --pred_data $OUTPUT_DIR/output_strap_full_0.txt \
    --output $OUTPUT_DIR/stats_0.txt

# top_p 0.6
python strap_many.py \
    --input $INPUT \
    --output $OUTPUT_DIR/output_strap_full_6.txt \
    --model_dir $MODEL_DIR \
    --batch_size 1 \
    --top_p_value 0.6

python utils/evaluate.py \
    --pred_data $OUTPUT_DIR/output_strap_full_6.txt \
    --output $OUTPUT_DIR/stats_6.txt

# top_p 0.9
python strap_many.py \
    --input $INPUT \
    --output $OUTPUT_DIR/output_strap_full_9.txt \
    --model_dir $MODEL_DIR \
    --batch_size 1 \
    --top_p_value 0.9

python utils/evaluate.py \
    --pred_data $OUTPUT_DIR/output_strap_full_9.txt \
    --output $OUTPUT_DIR/stats_9.txt
