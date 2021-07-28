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


'''
This file contains all methods to train a EBR model. The model architecture is hardcoded.

'''


LOG_FILE = './log_train_big.log'
logging.basicConfig(filename= LOG_FILE, filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())



def get_parser():
    parser = argparse.ArgumentParser(description='EBR training')
    parser.add_argument('-method',  required=True, type=str,choices=['ebr','cebr'], help='dataset')
    parser.add_argument('-sample_count', default=100, metavar='N', type=int, help='Number of sypothesis/sample')
    parser.add_argument('-batch_size', default=128, metavar='N', type=int, help='Training batch size')
    parser.add_argument('-energy_temp', default=1000, metavar='N', type=int, help='Softmax energy temperature')
    parser.add_argument('-margin_weight', default=10.0, type=float, help='Bleu margin weight')
    parser.add_argument('-mixing_p', '--p', default=0.7, type=float, help='Mixture of probability of gold and hypothesis as target')
    parser.add_argument('-path', default='./', required=True, type=str, help='path to data')
    parser.add_argument('-train', default='train.csv', required=True, type=str, help='filename of training data')
    parser.add_argument('-valid', default='valid.csv', required=True, type=str, help='filename of validation data')
    parser.add_argument('-test', default='test.csv', required=True, type=str, help='filename of test data')
    parser.add_argument('-save_dir', default='checkpoints', type=str, help='Checkpoint directory')

    return parser  

# Loading arguments 
parser = get_parser()
args = parser.parse_args()
sample_count = args.sample_count

# Creating save directory
save_dir=args.save_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if args.method=='ebr':
   bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.method=='cebr':
   bert_tokenizer = BertTokenizer.from_pretrained("multibert/tokenizer/filepath")

   





class BertLMModel(nn.Module) : 
    
    def __init__(self):
        super().__init__()
        if args.method=='ebr':
           self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True) 
        elif args.method=='cebr':
             self.bert_model = BertForMaskedLM.from_pretrained('multibert/filetunes/checkpoint_folder', output_hidden_states = True) 


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
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key),
                                         self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
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
    train_file = args.train 
    valid_file = args.valid
    test_file=args.test
    data_file = pd.read_csv(os.path.join(path, train_file))
    field_names = list(data_file.columns)
    data_fields = []
    for field_name in field_names:
        field_obj = None
        field_info = field_name.split('_')
        if len(field_info) == 2 and field_info[0] == 'bleu' : field_obj =  data.Field(sequential = False, use_vocab = False)
        else : field_obj = TextField()
        data_fields.append((field_name, field_obj))                                                 
    train, val,test = data.TabularDataset.splits(path=path, train=train_file, validation=valid_file, test=test_file, format='csv', fields=data_fields, skip_header=True)
    return train, val, test, field_names



max_src_in_batch, max_tgt_in_batch = 25000, 25000

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

def run_eval_bleu(epoch, data_iter, model, field_names):    

    '''
    sample_filename = "bleu-test/ebm.out.sys"
    target_filename = "bleu-test/ebm.out.ref"
    sample_file = open(sample_filename, "w+")
    target_file = open(target_filename, "w+")
    '''

    refs = []
    sys = [] 
    for i, batch in enumerate(data_iter):
        hypo_scores = torch.stack([model(getattr(batch, field).cuda()) for field in field_names[1:1 + sample_count]]).view(sample_count, -1).t() 
        target_texts = getattr(batch, 'target')
        _, max_idx = torch.max(hypo_scores, 1)
        hypo_texts = [getattr(batch, 'hypo_' + str(idx.item()))[it].tolist() for it, idx in enumerate(max_idx)]
        hypo_sents = get_sent_from_tensor(hypo_texts)
        target_sents = get_sent_from_tensor(target_texts.tolist())
        refs.extend(target_sents)
        sys.extend(hypo_sents) 
        
        '''
        batch_count = len(hypo_sents)
        for idx in range(batch_count):
            logging.info("Target = {}".format(target_sents[idx]))
            logging.info("Top Hypo = {}".format(hypo_sents[idx])) 
            sample_file.write(hypo_sents[idx] + '\n')
            target_file.write(target_sents[idx] + '\n')
        '''

    #sample_file.close() ; target_file.close()
    
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(sys, refs)
    bleu_score = bleu.score
    return bleu_score
                       

def conditional_run_eval_bleu(epoch, data_iter, model, field_names):    

    '''
    sample_filename = "bleu-test/ebm.out.sys"
    target_filename = "bleu-test/ebm.out.ref"
    sample_file = open(sample_filename, "w+")
    target_file = open(target_filename, "w+")
    '''

    refs = []
    sys = [] 
    for i, batch in enumerate(data_iter):
        hypo_scores = torch.stack([model(getattr(batch, field).cuda()) for field in field_names[1:1 + sample_count]]).view(sample_count, -1).t() 
        target_texts = getattr(batch, 'target')
        _, max_idx = torch.max(hypo_scores, 1)
        for it, idx in enumerate(max_idx):
            hypo_texts=[getattr(batch, 'hypo_' + str(idx.item()))[it].tolist()]
            hypo_sent=bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(hypo_texts[0],skip_special_tokens = False))        
            hypo_target= bert_tokenizer.decode(bert_tokenizer.encode(f.split('[SEP]')[1]),skip_special_tokens = True) 
            sys.extend(hypo_target) 
            
        for i in range(len(target_texts.tolist())):
            target_sent=bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(target_texts.tolist()[i],skip_special_tokens = False))        
            target= bert_tokenizer.decode(bert_tokenizer.encode(f.split('[SEP]')[1]),skip_special_tokens = True) 
            refs.extend(target)

              
        
        
        
        '''
        batch_count = len(hypo_sents)
        for idx in range(batch_count):
            logging.info("Target = {}".format(target_sents[idx]))
            logging.info("Top Hypo = {}".format(hypo_sents[idx])) 
            sample_file.write(hypo_sents[idx] + '\n')
            target_file.write(target_sents[idx] + '\n')
        '''

    #sample_file.close() ; target_file.close()
    
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(sys, refs)
    bleu_score = bleu.score
    return bleu_score



def run_epoch(epoch, data_iter, model, field_names, args, opt = None):
    '''
    Run one training epoch.  
    '''

    total_loss = 0.0
    count  = 0
    print_sum = True
    energy_temp = args.energy_temp
    margin_weight = args.margin_weight
    p = args.p 


    for i, batch in enumerate(data_iter):
        count += 1
        sample_scores = torch.stack([model(getattr(batch, field).cuda()) for field in field_names[:1 + sample_count]]).view(sample_count+1, -1).t()
        bleu_scores = torch.stack([getattr(batch, 'bleu_' + str(field)).cuda() for field in range(sample_count)]).view(sample_count, -1).t()

        batch_hypo_weights = F.softmax(sample_scores[:,1:]/energy_temp, dim = 1) 
        batch_hypo_index = torch.multinomial(batch_hypo_weights, num_samples=1)
        batch_target_index = torch.multinomial(batch_hypo_weights, num_samples=1)

        batch_sample_count = len(sample_scores)
        batch_target_mask = torch.rand((batch_sample_count,)).cuda()

        # mask to choose gold translation as target or sample a hypothesis as target
        batch_target_mask[batch_target_mask < p] = 0
        batch_target_mask[batch_target_mask > 0] = 1
        batch_target_mask = batch_target_mask.long().view(-1, 1)

        hypo_bleu = bleu_scores.gather(1, batch_hypo_index)
        target_bleu = bleu_scores.gather(1, batch_target_mask*batch_target_index)
        ideal_bleu = torch.ones_like(target_bleu)*100
        reverse_mask = 1 - batch_target_mask
        final_target_bleu = reverse_mask*ideal_bleu + batch_target_mask*target_bleu

        bleu_diff = (final_target_bleu-hypo_bleu).squeeze()
        loss_mask = hypo_bleu - final_target_bleu
        loss_mask[loss_mask < 0] = -1
        loss_mask[loss_mask > 0] = 1

        target_score = sample_scores.gather(1, batch_target_mask*(batch_target_index+1))*loss_mask
        hypo_score = sample_scores.gather(1, batch_hypo_index+1)*loss_mask
        
        target_score = target_score.squeeze()
        hypo_score = hypo_score.squeeze()
        
        loss = target_score - hypo_score + abs(bleu_diff)*margin_weight
        loss[loss<0] = 0

        loss += + 0.0001*(torch.sum(target_score**2) + torch.sum(hypo_score**2))
        loss = torch.sum(loss) + 1e-7
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() 
        if i %1000 == 1: logging.info("Epoch {} , batch {}, loss = {}".format(epoch, i,  total_loss/i))
        if print_sum and i %1000 == 1: 
            t_score = torch.sum(target_score).item()
            s_score = torch.sum(hypo_score).item()
            t_bleu = torch.sum(final_target_bleu).item()
            s_bleu = torch.sum(hypo_bleu).item()
            loss = loss.item()
            logging.info("Epoch {} , batch {}, target score = {}, sample score = {}, target_bleu  = {}, hypo_bleu = {} , loss = {}".format(epoch, i, t_score, s_score, t_bleu, s_bleu, loss))
    return total_loss/count



def train(args) :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.current_device())
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    model = BertLMModel()
    train, val,test, field_names = get_data(args)
    BATCH_SIZE= args.batch_size
    SAVE_EVERY = 4
    #devices = [0]
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device='cuda:0', repeat=False,
                            sort_key=lambda x: tuple([len(getattr(x, field)) for field in field_names[:-1]]), batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device='cuda:0', repeat=False, sort_key=lambda x: tuple([len(getattr(x, field)) for field in field_names[:-1]]), batch_size_fn=batch_size_fn, train=False)
    
    test_iter=  MyIterator(test, batch_size=BATCH_SIZE, device='cuda:0', repeat=False,
                            sort_key=lambda x: tuple([len(getattr(x, field)) for field in field_names[:-1]]), batch_size_fn=batch_size_fn, train=False)
    model_par = model 
    model_opt =torch.optim.Adam(model.classifier.parameters(), lr=0.01)

    epochs = 80
    max_bleu = -1
    for epoch in range(epochs):
        if torch.cuda.device_count()>1:
            print("Number of GPUS {}".format(torch.cuda.device_count()))
            model_par = nn.DataParallel(model_par)
        model_par.to(device)
        model_par.train()
        train_loss = run_epoch(epoch, (rebatch(b, field_names) for b in train_iter), model_par, field_names, args, model_opt) 
        
        # Calculate sacreBleu on validation set
        model_par.eval()
        if args.method == 'ebr':
           valid_bleu = run_eval_bleu(epoch, (rebatch(b, field_names) for b in valid_iter), model_par, field_names)
           test_bleu=run_eval_bleu(epoch, (rebatch(b, field_names) for b in test_iter), model_par, field_names)
        elif args.method == 'cebr':
             valid_bleu = conditional_run_eval_bleu(epoch, (rebatch(b, field_names) for b in valid_iter), model_par, field_names)
             test_bleu=conditional_run_eval_bleu(epoch, (rebatch(b, field_names) for b in test_iter), model_par, field_names)        
        logging.info("Epoch {} : Training loss = {}, Validation BLEU = {}, test bleu={}".format(epoch, train_loss, valid_bleu,test_bleu))
        if max_bleu == -1 or max_bleu < valid_bleu or epoch % SAVE_EVERY == 0:
            max_bleu = max(max_bleu, valid_bleu)
            model_path = os.path.join(args.save_dir, 'model_{0}_{1:.2f}.pt'.format(epoch, valid_bleu))
            torch.save({
            'epoch': epoch,
            'cls_state_dict': model.classifier.state_dict(),
            'loss': valid_bleu,
            }, model_path)


train(args)
