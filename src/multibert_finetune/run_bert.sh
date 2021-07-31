DIR='conditionedbert/'
OUTPUT_DIR='bert-sep'
LOGGING_DIR='logs'

mkdir -p $OUTPUT_DIR
mkdir -p $LOGGING_DIR

CUDA_LAUNCH_BLOCKING=1 python bert_using_trainerE.py \
    -path $DIR \
    -train $TRAIN_File \
    -valid $VALID_File \
    -output_dir $OUTPUT_DIR \
    -logging_dir $LOGGING_DIR
