
# Energy-Based Reranking: Improving Neural Machine Translation Using Energy-Based Models

We have introduced two types of Energy Based Reranking models (Marginal EBMs and Joint-EBMs)

In order to train the model, following instructions are necessary:

1. Download the data and preprocess using fairseq format.

2. Set up a base-NMT (in our case, it is a transformer) train the base-NMT with the preprocessed dataset.

3. Use the trained NMT for sample generation. (100 samples in our case) 

4. prepare the data suitable as input to the ebr. This requires the target, samples and bleu score of the samples.

5. Train the EBR model. This requires the BERT model as energy score generator along with the prepared samples from the last section.

we dont use gold data in any part of training, (-- mixing-p is 0), in case target data is required for any part of training the mixing-p argument can be set between 0 to 1.

**** src/run.sh should execute the entire process mentioned above,
**** under the fairseq folder final-ebr.py executes download script (i.e.-bash downlaod_data-iwdeen.sh --ebr)
**** + transformer training (fairseq-train ..) script + sample-generation (uses fairseq format..) script + data preparation ( executes a seperate python file dataprep-
**** ebr.py) script + ebr training script (python train_ebr.py) 

**P.S.**- For conditional ebr a finetuned multibert with the corresponding language pair is required.


![til](https://github.com/sumantakcs/ebr-nmt/blob/2d77c6112b808584c6b1f84c0e9d9e63814ae7b3/Presentation6.gif)
 
 
 
**For Marginal-EBM it does not require to finetune the BERT as it is bert-base un-cased model for conditional EBMs, for joint EBMs it is require to download the Multi-Bert model(https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) and finetune it with corresponding language pairs in [CLS]source[SEP]target format[SEP] before using it as the bert-based enrgy value generator.**

**This code requires fairseq, Please install fairseq and set it up before cloning this repo.**

<p>git clone https://github.com/pytorch/fairseq </p>
<p>cd fairseq </p>
<p>pip install --editable ./</p>





