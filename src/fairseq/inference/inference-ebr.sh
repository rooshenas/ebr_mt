##<> requires file_path
fairseq-generate '<data_path>' --gen-subset test --path '<NMT_checkpoint>' --nbest 100 --beam 100 --batch-size 64 --remove-bpe @@ | tee gentest.out

grep ^S gentest.out | cut -f2- > sourcetest1.txt
grep ^T gentest.out | cut -f2- > targettest1.txt
grep ^H gentest.out | cut -f3- > hypothesestest1.txt
python dataprep-ebr.py
python inf-ebr.py -path ./ -test testdata1.csv

