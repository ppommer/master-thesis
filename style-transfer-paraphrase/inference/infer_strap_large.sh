MODEL=large

MODEL_DIR=../models/strap_${MODEL}
OUTPUT_DIR=strap_${MODEL}

INPUT_SINGLE=../data/WNC/single_biased_test.txt
INPUT_MULTI=../data/WNC/multi_biased_test.txt

GOLD_SINGLE=../data/WNC/single_neutral_test.txt
GOLD_MULTI=../data/WNC/multi_neutral_test.txt

for i in 0 6 9; do

    echo output_strap_${MODEL}_single_${i} ...

    python ../strap_many.py \
        --input $INPUT_SINGLE \
        --output $OUTPUT_DIR/output_strap_${MODEL}_single_${i}.txt \
        --model_dir $MODEL_DIR \
        --batch_size 1 \
        --top_p_value 0.${i}

    echo stats_strap_${MODEL}_single_${i} ...

    python ../utils/evaluate.py \
        --pred_data $OUTPUT_DIR/output_strap_${MODEL}_single_${i}.txt \
        --output $OUTPUT_DIR/stats_strap_${MODEL}_single_${i}.txt \
        --gold_data $GOLD_SINGLE \
        --in_data $INPUT_SINGLE

    echo output_strap_${MODEL}_multi_${i} ...

    python ../strap_many.py \
        --input $INPUT_MULTI \
        --output $OUTPUT_DIR/output_strap_${MODEL}_multi_${i}.txt \
        --model_dir $MODEL_DIR \
        --batch_size 1 \
        --top_p_value 0.${i}

    echo stats_strap_${MODEL}_multi_${i} ...

    python ../utils/evaluate.py \
        --pred_data $OUTPUT_DIR/output_strap_${MODEL}_multi_${i}.txt \
        --output $OUTPUT_DIR/stats_strap_${MODEL}_multi_${i}.txt \
        --gold_data $GOLD_MULTI \
        --in_data $INPUT_MULTI
done
