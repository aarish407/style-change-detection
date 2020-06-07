import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle
from SplitIntoSentences import split_into_sentences
import json, os
import numpy as np 
import psutil
import gc
from pympler import muppy, summary
import tracemalloc

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = BertModel.from_pretrained('bert-base-cased')
if torch.cuda.is_available():
    model= model.cuda()
model.eval()

import joblib

with open('DocuNarrow.joblib', 'rb') as file_handle:
    clf_docu_narrow= joblib.load(file_handle)

with open('DocuWide.joblib', 'rb') as file_handle:
    clf_docu_wide= joblib.load(file_handle)

with open('ParaNarrow.joblib', 'rb') as file_handle:
    clf_para_narrow= joblib.load(file_handle)

with open('ParaWide.joblib', 'rb') as file_handle:
    clf_para_wide= joblib.load(file_handle)

def generate_sentence_embedding(sentence):
    marked_sentence= "[CLS] " + sentence + " [SEP]"
    tokenized_sentence= tokenizer.tokenize(marked_sentence)
    if len(tokenized_sentence) > 512: # truncate the sentence if it is longer than 512
        tokenized_sentence= tokenized_sentence[:512]
    
    indexed_tokens= tokenizer.convert_tokens_to_ids(tokenized_sentence)
    segment_ids= [1] * len(tokenized_sentence)

    token_tensor= torch.tensor([indexed_tokens])
    segment_tensor= torch.tensor([segment_ids])

    if torch.cuda.is_available():
        token_tensor= token_tensor.cuda()
        segment_tensor= segment_tensor.cuda()

    with torch.no_grad():
        encoded_layers, _= model(token_tensor, segment_tensor)
    
    token_embeddings= torch.stack(encoded_layers, dim= 0)
    token_embeddings= torch.squeeze(token_embeddings, dim= 1)
    token_embeddings= torch.sum(token_embeddings[-4:,:,:], dim= 0)
    sentence_embedding_sum= torch.sum(token_embeddings, dim= 0)

    # del marked_sentence
    # del tokenized_sentence
    # del indexed_tokens, segment_ids
    # del token_tensor
    # del segment_tensor
    # del encoded_layers
    # del token_embeddings
    # gc.collect()

    return sentence_embedding_sum


'''
read the generate_embddings_wide function to understand the logic, generate_embddings_narrow is the same but has all my debugging
print statements
'''

def generate_embeddings_narrow(corpora, path):
    # Add the classifier loading here

    pid= os.getpid()
    ps=psutil.Process(pid)

    iterator= iter(corpora)
    done_looping= False

    while not done_looping:
        tracemalloc.start(20)
        snapshot1 = tracemalloc.take_snapshot()
        try:
            document, document_id= next(iterator)
        except StopIteration:
            done_looping= True
            break

        if not document or not document_id:
            continue
        
        print('a. Entered the document for loop',psutil.virtual_memory().used/(1024*1024), ps.memory_info().vms/(1024*1024))
        current= psutil.virtual_memory().used/(1024*1024)
        document_embeddings= torch.zeros(768)
        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cuda()

        sentence_count= 0
        paragraphs_embeddings= []
        paragraphs= document.split('\n\n')
        
        previous_para_embeddings= None
        previous_para_length= None
        
        print('b. Before the paragraph for loop',psutil.virtual_memory().used/(1024*1024)-current)
        current= psutil.virtual_memory().used/(1024*1024)
        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)
            # print('after split into sentences',len(sentences),psutil.virtual_memory().used-current)
            # current= psutil.virtual_memory().used

            current_para_embeddings= torch.zeros(768)
            if torch.cuda.is_available():
                current_para_embeddings= current_para_embeddings.cuda()

            current_para_length= len(sentences)

            for sentence in sentences:
                sentence_count+=1 
                sentence_embedding= generate_sentence_embedding(sentence)         
                current_para_embeddings.add_(sentence_embedding)
                document_embeddings.add_(sentence_embedding)
                # del sentence_embedding, sentence
            # print('after sentence calculation',len(sentences),psutil.virtual_memory().used-current)
            # current= psutil.virtual_memory().used

            if previous_para_embeddings is not None:
                two_para_lengths= previous_para_length + current_para_length
                two_para_embeddings= (previous_para_embeddings + current_para_embeddings)/two_para_lengths
        
                paragraphs_embeddings.append(two_para_embeddings)            
            
            previous_para_embeddings = current_para_embeddings
            previous_para_length = current_para_length
            # del sentences
            # del paragraph
            # gc.collect()


        # del previous_para_embeddings, previous_para_length
        # del current_para_embeddings, current_para_length
        # del two_para_embeddings
        
        print('c. After the paragaph for loop',psutil.virtual_memory().used/(1024*1024)-current)
        current= psutil.virtual_memory().used/(1024*1024)
        
        paragraphs_embeddings= torch.stack(paragraphs_embeddings, dim=0)
        document_embeddings= document_embeddings/sentence_count
        document_embeddings= document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cpu()
            paragraphs_embeddings= paragraphs_embeddings.cpu()

        #### PREDICTIONS 

        print('d. before the document predicition classifier',psutil.virtual_memory().used/(1024*1024)-current)
        current= psutil.virtual_memory().used/(1024*1024)
        try:
            document_label= clf_docu_narrow.predict(document_embeddings)
        except:
            document_label= [0]
        print('e. after the document predicition classifier and before paragraph pred classifier',psutil.virtual_memory().used/(1024*1024)-current)
        current= psutil.virtual_memory().used/(1024*1024)
        
        try:
            paragraphs_labels= clf_para_narrow.predict(paragraphs_embeddings)
        except:
            paragraphs_labels= np.zeros(len(paragraphs)-1)
        paragraphs_labels= paragraphs_labels.astype(np.int32)
        print('f. after the paragraph predicition classifier',psutil.virtual_memory().used/(1024*1024)-current)
        current= psutil.virtual_memory().used/(1024*1024)
        
        solution= {
            'multi-author': document_label[0],
            'changes': paragraphs_labels.tolist()
        }

        print('h. after making solution dictionary',psutil.virtual_memory().used/(1024*1024)-current)
        current= psutil.virtual_memory().used/(1024*1024)


        file_name= path+'/solution-problem-'+document_id+'.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)
        
        # del document_embeddings, document_label
        # del paragraphs_embeddings, paragraphs_labels, paragraphs
        # del solution
        # del document 
        del paragraphs
        gc.collect()

        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)

        print("j. after saving the solution to: ", file_name,psutil.virtual_memory().used/(1024*1024)-current, ps.memory_info().vms/(1024*1024))
        current= psutil.virtual_memory().used/(1024*1024)
        
        print()
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        for stat in top_stats[:20]:
            print(stat)
        print()

        

            
def generate_embeddings_wide(corpora, path):
    # classifier loading
    for document, document_id in corpora:
        if not document or not document_id:
            continue
        
        ## read the file

        document_embeddings= torch.zeros(768)
        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cuda()

        sentence_count= 0
        paragraphs_embeddings= []
        paragraphs= document.split('\n\n')
        
        previous_para_embeddings= None
        previous_para_length= None

        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)

            current_para_embeddings= torch.zeros(768)
            if torch.cuda.is_available():
                current_para_embeddings= current_para_embeddings.cuda()

            current_para_length= len(sentences)

            for sentence in sentences:
                sentence_count+=1 
                sentence_embedding= generate_sentence_embedding(sentence)         
                current_para_embeddings.add_(sentence_embedding)
                document_embeddings.add_(sentence_embedding)
            
            if previous_para_embeddings is not None:
                two_para_lengths= previous_para_length + current_para_length
                two_para_embeddings= (previous_para_embeddings + current_para_embeddings)/two_para_lengths
        
                paragraphs_embeddings.append(two_para_embeddings)            

            previous_para_embeddings = current_para_embeddings
            previous_para_length = current_para_length
        
        # print('c. After the paragaph for loop')
        
        paragraphs_embeddings= torch.stack(paragraphs_embeddings, dim=0)
        document_embeddings= document_embeddings/sentence_count
        document_embeddings= document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cpu()
            paragraphs_embeddings= paragraphs_embeddings.cpu()

        #### PREDICTIONS 

        try:
            document_label= clf_docu_wide.predict(document_embeddings)
        except: 
            document_label= [0]

        try:
            paragraphs_labels= clf_para_wide.predict(paragraphs_embeddings)
        except:
            paragraphs_labels= np.zeros(len(paragraphs)-1)
        paragraphs_labels= paragraphs_labels.astype(np.int32)

        solution= {
            'multi-author': document_label[0],
            'changes': paragraphs_labels.tolist()
        }

        file_name= path+'/solution-problem-'+document_id+'.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
