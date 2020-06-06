import json
import glob

def load_dataset(path):
    input_txt= glob.glob(path+'/*.txt')

    dataset= []

    for i in range(len(input_txt)):
        document_id= (input_txt[i][len(path)+9:-4]) 

        dataset.append([])

        with open(input_txt[i], encoding="utf8") as file:
            dataset[-1].append(file.read())
        
        dataset[-1].append(document_id)

    return dataset
