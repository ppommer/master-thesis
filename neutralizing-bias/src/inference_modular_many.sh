#############
ITERATIONS=10
BASE_DIR=inference/modular_many
#############

COUNTER=1
TEST=bias_data/WNC/biased.word.test

for _ in $(seq $ITERATIONS); do
    INFERENCE_OUTPUT=$BASE_DIR/results_modular_$COUNTER.txt

    echo "**********"
    echo $COUNTER
    echo "inference.py INPUT: $TEST"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT"
    echo "**********"

    python joint/inference.py \
        --test $TEST \
        --inference_output $INFERENCE_OUTPUT \
        --working_dir $BASE_DIR \
        --checkpoint models/modular.ckpt \
        --activation_hidden \
        --bert_full_embeddings \
        --coverage --debias_weight 1.3 \
        --extra_features_top \
        --pointer_generator \
        --pre_enrich \
        --token_softmax \
        --test_batch_size 1

    python utils/generate_output.py \
        --in_file $INFERENCE_OUTPUT \
        --out_file $BASE_DIR/output_modular_$COUNTER.txt \
        --html_file $BASE_DIR/output_modular_$COUNTER.html

    COUNTER=$(( COUNTER + 1 ))

    if [ $COUNTER -le $ITERATIONS ]; then
        echo "**********"
        echo "prepare_next.py INPUT: $INFERENCE_OUTPUT"
        echo "prepare_next.py OUPUT: $TEST"
        echo "**********"

        TEST=$BASE_DIR/input.txt

        python utils/prepare_next.py \
            --input $INFERENCE_OUTPUT \
            --output $TEST
    fi
done

rm $BASE_DIR/input.txt
