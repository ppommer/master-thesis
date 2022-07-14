#############
ITERATIONS=10
BASE_DIR=inference/concurrent_many
#############

COUNTER=1
TEST=bias_data/WNC_edit/single_test.txt
TEST_MULTI=bias_data/WNC_edit/multi_test.txt

for _ in $(seq $ITERATIONS); do
    INFERENCE_OUTPUT=$BASE_DIR/results_concurrent_$COUNTER.txt

    echo "**********"
    echo $COUNTER
    echo "inference.py INPUT: $TEST"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT"
    echo "**********"

    python joint/inference.py \
        --test $TEST \
        --inference_output $INFERENCE_OUTPUT \
        --working_dir $BASE_DIR \
        --bert_encoder \
        --bert_full_embeddings \
        --coverage \
        --debias_checkpoint models/concurrent.ckpt \
        --debias_weight 1.3 \
        --no_tok_enrich \
        --pointer_generator \
        --test_batch_size 1

    python utils/generate_output.py \
        --in_file $INFERENCE_OUTPUT \
        --out_file $BASE_DIR/output_concurrent_$COUNTER.txt

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

    INFERENCE_OUTPUT_MULTI=${BASE_DIR}/multi/results_concurrent_multi_${COUNTER}.txt

    echo "**********"
    echo ${COUNTER}
    echo "inference.py INPUT: $TEST_MULTI"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT_MULTI"
    echo "**********"

    python joint/inference.py \
        --test ${TEST_MULTI} \
        --inference_output ${INFERENCE_OUTPUT_MULTI} \
        --working_dir ${BASE_DIR}/multi \
        --bert_encoder \
        --bert_full_embeddings \
        --coverage \
        --debias_checkpoint models/concurrent.ckpt \
        --debias_weight 1.3 \
        --no_tok_enrich \
        --pointer_generator \
        --test_batch_size 1

    python utils/generate_output.py \
        --in_file ${INFERENCE_OUTPUT_MULTI} \
        --out_file $BASE_DIR/multi/output_concurrent_multi_${COUNTER}.txt

    COUNTER=$(( COUNTER + 1 ))

    if [ $COUNTER -le $ITERATIONS ]; then
        echo "**********"
        echo "prepare_next.py INPUT: $INFERENCE_OUTPUT_MULTI"
        echo "prepare_next.py OUPUT: $TEST_MULTI"
        echo "**********"

        TEST_MULTI=${BASE_DIR}/multi/input.txt

        python utils/prepare_next.py \
            --input ${INFERENCE_OUTPUT_MULTI} \
            --output ${TEST_MULTI}
    fi
done

rm $BASE_DIR/input.txt
rm $BASE_DIR/multi/input.txt
