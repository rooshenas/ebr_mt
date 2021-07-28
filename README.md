We have introduced two types of Energy Based Reranking models (Marginal EBMs and Joint-EBMs)

code includes following steps-

1. Data downloadng+preprocessing

2. Transformer training

3. sample generation from transformer

4. make data ready for EBR input

5. Train the model


                                                       ![alt text](https://github.com/sumantakcs/ebr-nmt/blob/2d77c6112b808584c6b1f84c0e9d9e63814ae7b3/Presentation6.gif"ebr information flow")
**For Marginal-EBM it does not require to finetune the BERT as it is bert-base un-cased model for conditional EBMs, for joint EBMs it is require to download the Multi-Bert model(https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) and finetune it with corresponding language pairs in [CLS]source[SEP]target format[SEP] before using it as the bert-based enrgy value generator.**



