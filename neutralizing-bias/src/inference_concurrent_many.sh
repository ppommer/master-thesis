#############
ITERATIONS=10
BASE_DIR=/home/ppommer/repos/master-thesis/neutralizing-bias/src
#############

COUNTER=1
TEST=$BASE_DIR/bias_data/WNC/biased.word.test

for _ in $(seq $ITERATIONS); do
    INFERENCE_OUTPUT=$BASE_DIR/inference/concurrent/iter_$ITERATIONS/results_concurrent_$COUNTER.txt

    echo "**********"
    echo $COUNTER
    echo "inference.py INPUT: $TEST"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT"
    echo "**********"

    python joint/inference.py \
        --test $TEST \
        --inference_output $INFERENCE_OUTPUT \
        --bert_encoder \
        --bert_full_embeddings \
        --coverage \
        --debias_checkpoint $BASE_DIR/models/concurrent.ckpt \
        --debias_weight 1.3 \
        --no_tok_enrich \
        --pointer_generator \
        --working_dir inference_concurrent/

    COUNTER=$(( COUNTER + 1 ))
    TEST=$BASE_DIR/inference/concurrent/iter_$ITERATIONS/input_concurrent_$COUNTER.txt

    if [ $COUNTER -le $ITERATIONS ]; then
        echo "**********"
        echo "prepare_next.py INPUT: $INFERENCE_OUTPUT"
        echo "prepare_next.py OUPUT: $TEST"
        echo "**********"

        python prepare_next.py \
            --input $INFERENCE_OUTPUT \
            --output $TEST
    fi
done
