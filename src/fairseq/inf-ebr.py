import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from pytorch_transformers import *
import sacrebleu
from torchtext import data, datasets
from torchtext.vocab import Vectors
import pandas as pd
import logging
import argparse
from subprocess import PIPE, run




bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_parser():
    parser = argparse.ArgumentParser(description='EBR training')
    parser.add_argument('-sample_count', default=100, metavar='N', type=int, help='Number of sypothesis/sample')
    parser.add_argument('-batch_size', default=100, metavar='N', type=int, help='Training batch size')
    #parser.add_argument('-energy_temp', default=1000, metavar='N', type=int, help='Softmax energy temperature')
    #parser.add_argument('-margin_weight', default=10.0, type=float, help='Bleu margin weight')
    #parser.add_argument('-mixing_p', '--p', default=0.0, type=float, help='Mixture of probability of gold and hypothesis as target')
    parser.add_argument('-path', default='./', required=True, type=str, help='path to data')
    #parser.add_argument('-train', default='train.csv', required=True, type=str, help='filename of training data')
    #parser.add_argument('-valid', default='valid.csv', required=True, type=str, help='filename of validation data')
    parser.add_argument('-test', default='test.csv', required=True, type=str, help='filename of test data')
    #parser.add_argument('-save_dir', default='checkpoints', type=str, help='Checkpoint directory')

    return parser  


# Loading arguments 
parser = get_parser()
args = parser.parse_args()
sample_count = args.sample_count

# Creating save directory
#save_dir=args.save_dir
#if not os.path.exists(save_dir):
#    os.mkdir(save_dir)



class BertLMModel(nn.Module) : 
    
    def __init__(self):
        super().__init__()
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True) 
        self.bert_model.eval()
        self.classifier = nn.Sequential(nn.Linear(768, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 1))
    
    def forward(self, hypo_sample):
        with  torch.no_grad():
            _, sample_hidden = self.bert_model(hypo_sample)
        sample_scores = self.classifier(sample_hidden[-1])
        sample_scores = torch.mean(sample_scores, axis = 1) 
        return sample_scores



class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, batch_obj, field_names):
        for field in field_names:
            setattr(self, field, getattr(batch_obj, field))

class MyIterator(data.Iterator):
    def create_batches(self):
        self.batches = []
        for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
            self.batches.append(sorted(b, key=self.sort_key))



def rebatch(batch, field_names):
    """
    Fix order in torchtext to match ours
    """
    return Batch(batch, field_names)

def text_to_tensor(text):
    tokens = bert_tokenizer.tokenize(text)
    tensor_input = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
    return tensor_input

class TextField(data.Field):
    def __init__(self, *args):
        super().__init__(self, tokenize = bert_tokenizer.tokenize, use_vocab = False, pad_token= bert_tokenizer.pad_token, init_token = bert_tokenizer.bos_token, eos_token = bert_tokenizer.eos_token)

    def numericalize(self, arr, device=None):
        arr = [bert_tokenizer.convert_tokens_to_ids(sent) for sent in arr]
        arr_tensor = torch.tensor(arr, device = device)
        return arr_tensor
     
def get_data(args) :
    path = args.path 
    #train_file = args.train 
    #valid_file = args.valid
    test_file=args.test
    data_file = pd.read_csv(os.path.join(path,test_file))
    field_names = list(data_file.columns)
    data_fields = []
    for field_name in field_names:
        field_obj = None
        field_info = field_name.split('_')
        if len(field_info) == 2 and field_info[0] == 'bleu' : field_obj =  data.Field(sequential = False, use_vocab = False)
        else : field_obj = TextField()
        data_fields.append((field_name, field_obj))                                                 
    test = data.TabularDataset(path=os.path.join(path,test_file), format='csv', fields=data_fields, skip_header=True)
    print('test#########################',test)
    return test, field_names



#max_src_in_batch, max_tgt_in_batch = 25000, 25000

def batch_size_fn(new, count, size_so_far, target_only_batching = True):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch
    global max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    total_src_in_batch = 0
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.target))
    tgt_elements = count * max_tgt_in_batch

    if not target_only_batching :
        for i in range(sample_count) :
            max_src_in_batch = max(max_src_in_batch, len(getattr(new, 'hypo_' + str(i))))
            total_src_in_batch += len(getattr(new, 'hypo_' + str(i)))
        max_src_in_batch = max(max_src_in_batch, total_src_in_batch//sample_count)
        src_elements = count * max_src_in_batch
        return max(src_elements, tgt_elements)

    return tgt_elements



def get_sent_from_tensor(tokens):
    return_sent = [bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(sent ,skip_special_tokens = True)) for sent in tokens]
    return return_sent

def output(data_iter, model, field_names):    
    
    
    '''
    sample_filename = "bleu-test/ebm.out.sys"
    target_filename = "bleu-test/ebm.out.ref"
    sample_file = open(sample_filename, "w+")
    target_file = open(target_filename, "w+")
    '''     
    data = pd.DataFrame()
    k=[]
    l=[]

    for i, batch in enumerate(data_iter):
        hypo_scores = torch.stack([model(getattr(batch, field)) for field in field_names[1:1 + sample_count]]).view(sample_count, -1).t() 
        target_texts = getattr(batch, 'target')
        _, max_idx = torch.max(hypo_scores, 1)
        hypo_texts = [getattr(batch, 'hypo_' + str(idx.item()))[it].tolist() for it, idx in enumerate(max_idx)]
        hypo_sents = get_sent_from_tensor(hypo_texts)
        target_sents = get_sent_from_tensor(target_texts.tolist())
        for i in range (len(target_sents)):
            k.append(hypo_sents[i])
            l.append(target_sents[i])
    data['target']=l
    data['Cond-EBR-output']=k
    p=data.to_csv('inferred_output-wmtdeen14',index=False)    

     
        
    return p             
def inference(args) :
    device_id = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device',device)
    print('Using device:', device)
    
    if device.type == 'cuda':
       print(torch.cuda.get_device_name(0))
       print('Memory Usage:')
       print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
       print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    model = BertLMModel()
    model.to(device)
    ckpt=torch.load('models/WMT-DE-EN14/checkpoints/wmtebrdeen.pt', map_location=device)
    model.classifier.load_state_dict(ckpt['cls_state_dict'])
    test, field_names = get_data(args)
    BATCH_SIZE= args.batch_size
    test_iter=  MyIterator(test, batch_size=BATCH_SIZE, repeat=False,device=device,
                            sort_key=lambda x: tuple([len(getattr(x, field)) for field in field_names[:-1]]), batch_size_fn=batch_size_fn, train=False)
    test_bleu=output((rebatch(b, field_names) for b in test_iter), model, field_names)
    return test_bleu
        

inference(args)
