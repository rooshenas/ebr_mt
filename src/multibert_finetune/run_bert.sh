DIR='conditionedbert/'
OUTPUT_DIR='bert-sep'
LOGGING_DIR='logs'
TRAIN_DIR='conditionedbert/combined_sentences_train-sep.txt'
VALID_DIR='conditionedbert/combined_sentences_valid-sep.txt'

mkdir -p $OUTPUT_DIR
mkdir -p $LOGGING_DIR

CUDA_LAUNCH_BLOCKING=1 python bert_using_trainerE.py \
    -path $DIR \
    -train $TRAIN_DIR \
    -valid $VALID_DIR \
    -output_dir $OUTPUT_DIR \
    -logging_dir $LOGGING_DIR
