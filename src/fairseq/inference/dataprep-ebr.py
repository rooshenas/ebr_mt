import sacrebleu
import numpy as np
import time

start_time = time.time()

source=open(r'sourcetest1.txt' ,"r",encoding='utf-8').readlines()
target=open(r'targettest1.txt', "r",encoding='utf-8').readlines()
hypothesis=open(r'hypothesestest1.txt', "r",encoding='utf-8').readlines()
print(hypothesis)
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
    print('tgt sent_map',tgt_sent_map)
    #target_list.append("[CLS]"+src_sent_original+"[SEP]"+tgt_sent_original)
    target_list.append(tgt_sent_original)
    #tgt_sent=".".join(tgt_sent.split('.')[1:])
    refs = [[tgt_sent_original]]
    curr_sample = hypothesis[j*sample_count:(j+1)*sample_count]
    for i in range(len(curr_sample)):
        curr_sample[i]=curr_sample[i].strip()
        curr_sample_map=replace_all(curr_sample[i],d)
        print('current_sample_map',curr_sample[i])
        #sample_list[i].append("[CLS]"+src_sent_original+"[SEP]"+curr_sample[i]) 
        sample_list[i].append(curr_sample[i]) 
        sys=[curr_sample[i]]
        # #print('sys',type(sys))
        # #print('ref',refs)
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
    


data.to_csv("testdata1.csv", index = False,encoding="utf-8")
print("--- %s seconds ---" % (time.time() - start_time))
    


