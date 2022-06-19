#############
ITERATIONS=10
BASE_DIR=/home/ppommer/repos/master-thesis/neutralizing-bias/src
#############

COUNTER=1
TEST=$BASE_DIR/bias_data/WNC/biased.word.test

for _ in $(seq $ITERATIONS); do
    INFERENCE_OUTPUT=$BASE_DIR/inference/modular/iter_$ITERATIONS/results_modular_$COUNTER.txt

    echo "**********"
    echo $COUNTER
    echo "inference.py INPUT: $TEST"
    echo "inference.py OUTPUT: $INFERENCE_OUTPUT"
    echo "**********"

    python joint/inference.py \
        --test $TEST \
        --inference_output $INFERENCE_OUTPUT \
        --checkpoint $BASE_DIR/models/modular.ckpt \
        --activation_hidden \
        --bert_full_embeddings \
        --coverage --debias_weight 1.3 \
        --extra_features_top \
        --pointer_generator \
        --pre_enrich \
        --token_softmax \
        --working_dir inference_modular/

    COUNTER=$(( COUNTER + 1 ))
    TEST=$BASE_DIR/inference/modular/iter_$ITERATIONS/input_modular_$COUNTER.txt

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
