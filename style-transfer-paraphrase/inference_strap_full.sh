##########
MODEL=full
##########

MODEL_DIR=style_paraphrase/models/WNC_$MODEL
OUTPUT_DIR=inference/strap_$MODEL

INPUT_SINGLE=datasets/WNC/singleword_biased_test.txt
INPUT_MULTI=datasets/WNC/multiword_biased_test.txt
GOLD_SINGLE=datasets/WNC/singleword_neutral_test.txt
GOLD_MULTI=datasets/WNC/multiword_neutral_test.txt

for i in 0 6 9; do
    python strap_many.py \
        --input $INPUT_SINGLE \
        --output $OUTPUT_DIR/output_strap_$MODEL_single_$i.txt \
        --model_dir $MODEL_DIR \
        --batch_size 1 \
        --top_p_value 0.$i

    python utils/evaluate.py \
        --pred_data $OUTPUT_DIR/output_strap_$MODEL_single_$i.txt \
        --output $OUTPUT_DIR/stats_single_$i.txt
        --gold_data $GOLD_SINGLE \
        --in_data $INPUT_SINGLE

    python strap_many.py \
        --input $INPUT_MULTI \
        --output $OUTPUT_DIR/output_strap_$MODEL_multi_$i.txt \
        --model_dir $MODEL_DIR \
        --batch_size 1 \
        --top_p_value 0.$i

    python utils/evaluate.py \
        --pred_data $OUTPUT_DIR/output_strap_$MODEL_multi_$i.txt \
        --output $OUTPUT_DIR/stats_multi_$i.txt
        --gold_data $GOLD_MULTI \
        --in_data $INPUT_MULTI
done
