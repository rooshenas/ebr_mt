#!/bin/sh

cd fairseq
python final-ebr.py -method ebr -dataset iwdeen

#method contains ebr,cebr
# cebr method requires bert finetune for corresponding dataset.
# dataset choices are 'iwdeen', 'wmtdeen', 'wmtende'

