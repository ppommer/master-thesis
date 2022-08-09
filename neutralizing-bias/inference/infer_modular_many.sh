BASE_DIR=modular_many
MODEL_DIR=../models/modular.ckpt

TEST=../data/WNC/single_test.txt
TEST_MULTI=../data/WNC/multi_test.txt

ITERATIONS=10
COUNTER=1

for _ in $(seq $ITERATIONS); do
    INFERENCE_OUTPUT=$BASE_DIR/results_modular_$COUNTER.txt

    echo "**********"
    echo $COUNTER
    echo "inference.py INPUT: $TEST"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT"
    echo "**********"

    python ../joint/inference.py \
        --test $TEST \
        --inference_output $INFERENCE_OUTPUT \
        --working_dir $BASE_DIR \
        --checkpoint $MODEL_DIR \
        --activation_hidden \
        --bert_full_embeddings \
        --coverage --debias_weight 1.3 \
        --extra_features_top \
        --pointer_generator \
        --pre_enrich \
        --token_softmax \
        --test_batch_size 1

    python ../utils/generate_output.py \
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

        python ../utils/prepare_next.py \
            --input $INFERENCE_OUTPUT \
            --output $TEST
    fi

    INFERENCE_OUTPUT_MULTI=${BASE_DIR}/multi/results_modular_multi_${COUNTER}.txt

    echo "**********"
    echo ${COUNTER}
    echo "inference.py INPUT: $TEST_MULTI"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT_MULTI"
    echo "**********"

    python ../joint/inference.py \
        --test ${TEST_MULTI} \
        --inference_output ${INFERENCE_OUTPUT_MULTI} \
        --working_dir ${BASE_DIR}/multi \
        --checkpoint MODEL_DIR \
        --activation_hidden \
        --bert_full_embeddings \
        --coverage --debias_weight 1.3 \
        --extra_features_top \
        --pointer_generator \
        --pre_enrich \
        --token_softmax \
        --test_batch_size 1

    python ../utils/generate_output.py \
        --in_file ${INFERENCE_OUTPUT_MULTI} \
        --out_file ${BASE_DIR}/multi/output_modular_multi_${COUNTER}.txt \
        --html_file ${BASE_DIR}/multi/output_modular_multi_${COUNTER}.html

    COUNTER=$(( COUNTER + 1 ))

    if [ $COUNTER -le $ITERATIONS ]; then
        echo "**********"
        echo "prepare_next.py INPUT: $INFERENCE_OUTPUT_MULTI"
        echo "prepare_next.py OUPUT: $TEST_MULTI"
        echo "**********"

        TEST_MULTI=${BASE_DIR}/multi/input.txt

        python ../utils/prepare_next.py \
            --input ${INFERENCE_OUTPUT_MULTI} \
            --output ${TEST_MULTI}
    fi
done

rm ${BASE_DIR}/input.txt
rm ${BASE_DIR}/multi/input.txt
