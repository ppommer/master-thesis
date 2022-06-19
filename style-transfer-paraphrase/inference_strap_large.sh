DATASET=WNC_large

MODEL_DIR=style_paraphrase/models/$DATASET
INPUT_DIR=datasets/WNC/$DATASET
OUTPUT_DIR=inference/$DATASET

# top_p 0.0
python -m strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value 0.0 \
    --input $INPUT_DIR/test.txt \
    --output $OUTPUT_DIR/output_strap_large_0.txt

# top_p 0.6
python -m strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value 0.6 \
    --input $INPUT_DIR/test.txt \
    --output $OUTPUT_DIR/output_strap_large_6.txt

# top_p 0.9
python -m strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value 0.9 \
    --input $INPUT_DIR/test.txt \
    --output $OUTPUT_DIR/output_strap_large_9.txt
