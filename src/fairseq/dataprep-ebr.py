import sacrebleu
import numpy as np
import time
import argparse

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-source', type=str, help='source file name')

parser.add_argument('-target', type=str, help='target file name')

parser.add_argument('-hypo', type=str, help='hypothesis filename')

parser.add_argument('-output', type=str, help='output name')
args = parser.parse_args()

source=open(args.source ,"r",encoding='utf-8').readlines()
target=open(args.target , "r",encoding='utf-8').readlines()
hypothesis=open(args.hypo , "r",encoding='utf-8').readlines()
d={"&quot;":'"',"&apos;":"'","&apos;s":"'s","&apos;t":"'t","&apos;re":"'re","&apos;ve":"'ve","&apos;m":"'m","&apos;ll":"'ll","&apos;d":"'d"}

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
sample_count=100
sample_list = [[] for i in range(sample_count)]
bleu_list = [[] for i in range(sample_count)]
target_list = []
splitter=len(hypothesis)//sample_count
for j in range(splitter):
    print(j)
    src_sent_original= source[j].strip()
    src_sent_map=replace_all(src_sent_original,d)
    tgt_sent_original = target[j].strip()
    tgt_sent_map=replace_all(tgt_sent_original,d)
    target_list.append(tgt_sent_original)
    refs = [[tgt_sent_original]]
    curr_sample = hypothesis[j*sample_count:(j+1)*sample_count]
    for i in range(len(curr_sample)):
        curr_sample[i]=curr_sample[i].strip()
        curr_sample_map=replace_all(curr_sample[i],d)
        sample_list[i].append(curr_sample[i]) 
        sys=[curr_sample[i]]
        bleu=sacrebleu.corpus_bleu(sys, refs)
        bleu_score = int(np.floor(bleu.score))
        bleu_list[i].append(bleu_score) 
import pandas as pd
data = pd.DataFrame()
data['target'] = target_list

for i in range(sample_count):
    index = 'hypo_' + str(i)
    data[index] = sample_list[i]
#data['bleu']=bleu_list
for i in range(sample_count):
    index = 'bleu_' + str(i)
    data[index] = bleu_list[i]
    


data.to_csv("{}.csv".format(args.output), index = False,encoding="utf-8")
print("--- %s seconds ---" % (time.time() - start_time))
    


