import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle
from SplitIntoSentences import split_into_sentences
import json
import numpy as np 

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

print('i. Before model is loaded')
model = BertModel.from_pretrained('bert-base-cased')
if torch.cuda.is_available():
    model= model.cuda()
model.eval()
print('ii. after model is loaded')

import joblib

print('iii. before docunarrow is loaded')
with open('DocuNarrow.joblib', 'rb') as file_handle:
    clf_docu_narrow= joblib.load(file_handle)
print('iv. after docunarrow is loaded and before docuwide')

with open('DocuWide.joblib', 'rb') as file_handle:
    clf_docu_wide= joblib.load(file_handle)
print('v. after docuwide is loaded and before paranarrow')

with open('ParaNarrow.joblib', 'rb') as file_handle:
    clf_para_narrow= joblib.load(file_handle)
print('vi. after paranarrow is loaded and before parawide')

with open('ParaWide.joblib', 'rb') as file_handle:
    clf_para_wide= joblib.load(file_handle)
print('vii. after parawide is loaded')

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

    del token_tensor
    del segment_tensor
    del encoded_layers
    del token_embeddings

    return sentence_embedding_sum

def generate_embeddings_narrow(corpora, path):
    for document, document_id in corpora:
        if not document or not document_id:
            continue
        
        print('a. Entered the document for loop')
        document_embeddings= torch.zeros(768)
        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cuda()

        sentence_count= 0
        paragraphs_embeddings= []
        paragraphs= document.split('\n\n')
        
        previous_para_embeddings= None
        previous_para_length= None
        
        print('b. Before the paragraph for loop')
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
        
        print('c. After the paragaph for loop')
        
        paragraphs_embeddings= torch.stack(paragraphs_embeddings, dim=0)
        document_embeddings= document_embeddings/sentence_count
        document_embeddings= document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cpu()
            paragraphs_embeddings= paragraphs_embeddings.cpu()

        #### PREDICTIONS 

        print('d. before the document predicition classifier')
        document_label= clf_docu_narrow.predict(document_embeddings)
        print('e. after the document predicition classifier and before paragraph pred classifier')
        paragraphs_labels= clf_para_narrow.predict(paragraphs_embeddings)
        print('f. after the paragraph predicition classifier')
        paragraphs_labels= paragraphs_labels.astype(np.int32)

        print('g. before making solution dictionary')
        solution= {
            'multi-author': document_label[0],
            'changes': paragraphs_labels.tolist()
        }
        print('h. after making solution dictionary')


        file_name= path+'/solution-problem-'+document_id+'.json'
        print("i. Before saving the solution to: ", file_name)
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)
        print("j. after saving the solution to: ", file_name)
        print()
        

            
def generate_embeddings_wide(corpora, path):
    for document, document_id in corpora:
        print('a. Entered the document for loop')
        if not document or not document_id:
            continue
        
        document_embeddings= torch.zeros(768)
        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cuda()

        sentence_count= 0
        paragraphs_embeddings= []
        paragraphs= document.split('\n\n')
        
        previous_para_embeddings= None
        previous_para_length= None

        print('b. Before the paragraph for loop')
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
        
        print('c. After the paragaph for loop')
        
        paragraphs_embeddings= torch.stack(paragraphs_embeddings, dim=0)
        document_embeddings= document_embeddings/sentence_count
        document_embeddings= document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings= document_embeddings.cpu()
            paragraphs_embeddings= paragraphs_embeddings.cpu()

        #### PREDICTIONS 

        print('d. before the document predicition classifier')
        document_label= clf_docu_wide.predict(document_embeddings)
        print('e. after the document predicition classifier and before paragraph pred classifier')
        paragraphs_labels= clf_para_wide.predict(paragraphs_embeddings)
        print('f. after the paragraph predicition classifier')
        paragraphs_labels= paragraphs_labels.astype(np.int32)

        print('g. before making solution dictionary')
        solution= {
            'multi-author': document_label[0],
            'changes': paragraphs_labels.tolist()
        }
        print('h. after making solution dictionary')

        file_name= path+'/solution-problem-'+document_id+'.json'
        print("i. Before saving the solution to: ", file_name)
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)

        print("j. after saving the solution to: ", file_name)
        print()
        

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()