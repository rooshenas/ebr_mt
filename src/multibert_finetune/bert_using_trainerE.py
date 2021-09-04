import transformers
import torch
import pandas as pd
import os
from transformers import AdamW
import argparse
import math
import codecs

codecs.register_error('strict', codecs.lookup_error('surrogateescape'))

from transformers import BertTokenizer,BertModel,BertForMaskedLM,Trainer,TrainingArguments,LineByLineTextDataset,TextDataset,DataCollatorForLanguageModeling

from transformers import AutoTokenizer, AutoModelWithLMHead

#tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

tokenizer = BertTokenizer.from_pretrained('path/multibert/token/')


def get_parser():
	parser = argparse.ArgumentParser(description='BERT training')
	parser.add_argument('-batch_size', default=128, metavar='N', type=int, help='Training batch size')
	parser.add_argument('-path', default='./', required=True, type=str, help='path to data')
	parser.add_argument('-train', default='combined.txt', required=True, type=str, help='filename of training data')
	parser.add_argument('-valid', default='combined.txt', required=True, type=str, help='filename of validation data')
	parser.add_argument('-output_dir', default='results', type=str, help='Output directory')
	parser.add_argument('-logging_dir', default='logs', type=str, help='Logging directory')

	return parser  

# Loading arguments 
parser = get_parser()
args = parser.parse_args()
#print(args.train)

# Creating directory
output_dir=args.output_dir
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

logging_dir=args.logging_dir
if not os.path.exists(logging_dir):
	os.mkdir(logging_dir)


def encode(comb_batch, tokenizer):
	encoded_inputs = tokenizer(comb_batch, return_tensors='pt', padding=True, truncation=True)
	return encoded_inputs

train_dataset = TextDataset(
			tokenizer=tokenizer, file_path=args.train, block_size=512
		)

valid_dataset = TextDataset(
			tokenizer=tokenizer, file_path=args.valid, block_size=512
		)
data_collator = DataCollatorForLanguageModeling(
			tokenizer=tokenizer, mlm='mlm', mlm_probability=0.15
		)
output_eval_file=  open("c_bert_eval.txt", "a",encoding="utf-8")

def train(train_dataset):
    model = BertForMaskedLM.from_pretrained("path/multibert/checkpoint")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.train()
    training_args = TrainingArguments(
		output_dir=args.output_dir,          # output directory
		num_train_epochs=20,              # total # of training epochs
		per_device_train_batch_size=2,  # batch size per device during training
		per_device_eval_batch_size=2,   # batch size for evaluation
		warmup_steps=500,                # number of warmup steps for learning rate scheduler
		weight_decay=0.01,               # strength of weight decay
		logging_dir=args.logging_dir,            # directory for storing logs
		do_train=True,
		do_eval=True
	)
    trainer = Trainer(
		model=model,                         
		args=training_args,      
		data_collator=data_collator,            
		train_dataset=train_dataset, 
		eval_dataset=valid_dataset)
    trainer.train()
    eval_output = trainer.evaluate()
    for key in sorted(eval_output.keys()):
        output_eval_file.write("%s = %s\n" % (key, str(eval_output[key])))
    trainer.save_model()

train(train_dataset)

