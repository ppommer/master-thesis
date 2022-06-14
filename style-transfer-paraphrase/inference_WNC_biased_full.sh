DATASET=WNC_biased_full

MODEL_DIR=style_paraphrase/models/$DATASET
INPUT_DIR=datasets/WNC/$DATASET
OUTPUT_DIR=inference/$DATASET

# top_p 0.0
python strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value 0.0 \
    --input $INPUT_DIR/test.txt \
    --output $OUTPUT_DIR/output_0.txt

# top_p 0.6
python strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value 0.6 \
    --input $INPUT_DIR/test.txt \
    --output $OUTPUT_DIR/output_6.txt

# top_p 0.9
python strap_many.py \
    --batch_size 64 \
    --model_dir $MODEL_DIR \
    --top_p_value 0.9 \
    --input $INPUT_DIR/test.txt \
    --output $OUTPUT_DIR/output_9.txt
