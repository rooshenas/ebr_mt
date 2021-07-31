#!/bin/bash

echo 'downloading data+preprocess'+feeding to transformer+data preparation for ebr'

cd examples/translation  
bash prepare-wmt14en2de.sh --icml17
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/wmt14_en_de
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_de \
    --workers 20

echo 'feeding to transformer'

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/wmt14_en_de \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-last-checkpoints \
    --max-epoch 70

echo 'sample generation for train validation and test dataset'

fairseq-generate 'data-bin/wmt14_en_de' \
--gen-subset train --path 'checkpoints/checkpoint_best.pt' \
--nbest 100 --beam 100 --batch-size 64 --remove-bpe @@  | tee wmtdeen.out


grep ^S wmtdeen.out | cut -f2- > train-source.txt
grep ^T wmtdeen.out | cut -f2- > train-target.txt
grep ^H wmtdeen.out | cut -f3- > train-hypo.txt

fairseq-generate 'data-bin/wmt14_en_de' \
--gen-subset valid --path 'checkpoints/checkpoint_best.pt' \
--nbest 100 --beam 100 --batch-size 64 --remove-bpe @@  | tee wmtdeen.out


grep ^S wmtdeen.out | cut -f2- > val-source.txt
grep ^T wmtdeen.out | cut -f2- > val-target.txt
grep ^H wmtdeen.out | cut -f3- > val-hypo.txt

fairseq-generate 'data-bin/wmt14_en_de' \
--gen-subset test --path 'checkpoints/checkpoint_best.pt' \
--nbest 100 --beam 100 --batch-size 64 --remove-bpe @@  | tee wmtdeen.out


grep ^S wmtdeen.out | cut -f2- > test-source.txt
grep ^T wmtdeen.out | cut -f2- > test-target.txt
grep ^H wmtdeen.out | cut -f3- > test-hypo.txt


echo 'data preparation for EBR'

if [ "$1" == "--ebr" ]; then
    python dataprep-ebr.py -source train-source.txt -target train-target.txt -hypo train-hypo.txt -output train-ebr
    python dataprep-ebr.py -source val-source.txt -target val-target.txt -hypo val-hypo.txt -output val-ebr
    python dataprep-ebr.py -source test-source.txt -target test-target.txt -hypo test-hypo.txt -output test-ebr
else
    python dataprep.py -source train-source.txt -target train-target.txt -hypo train-hypo.txt -output train-cebr
    python dataprep.py -source val-source.txt -target val-target.txt -hypo val-hypo.txt -output val-cebr
    python dataprep.py -source test-source.txt -target test-target.txt -hypo test-hypo.txt -output test-cebr

echo 'Ready for EBR training'























