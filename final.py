import sys
import pickle
import os

print('1. reached Python file')
from LoadDataset import load_dataset
from GenerateEmbeddings import generate_embeddings_narrow, generate_embeddings_wide

# TODO: Change this later

# Initialize two directories
input_dir= sys.argv[1]
output_dir= sys.argv[2]


# Read the two datasets
input_path_narrow= input_dir + '/dataset-narrow'
print('2. Before Dataset narrow load')
dataset_narrow= load_dataset(input_path_narrow)
print('3. After Dataset narrow load ',len(dataset_narrow))
dataset_narrow.sort(key = lambda x: int(x[1]))

input_path_wide= input_dir + '/dataset-wide'
print('4. Before Dataset wide load')
dataset_wide= load_dataset(input_path_wide)
print('5. After Dataset wide load ',len(dataset_wide))
dataset_wide.sort(key = lambda x: int(x[1]))

# Create two subfolders within output dir, dataset-narrow and dataset-wide
output_path_narrow= output_dir + '/dataset-narrow'
output_path_wide= output_dir + '/dataset-wide'

print('6. Before mkdir of narrow')
try:
    os.mkdir(output_path_narrow)
    print('7. after mkdir of narrow')

except OSError:
    print('7. OS Error in mkdir of narrow')

print('8. Before mkdir of narrow')
try:
    os.mkdir(output_path_wide)
    print('9. after mkdir of wide')
except OSError:
    print('9. OS Error in mkdir of wide')


# print(dataset_narrow[1060:1070][1])
# print(dataset_wide[1060:1070][1])
# Generate Embeddings and predict
print('10. Before generate embeddings narrow')
generate_embeddings_narrow(dataset_narrow, output_path_narrow)
print('11. after generate embeddings narrow and before wide')
generate_embeddings_wide(dataset_wide, output_path_wide)
print('12. after generate embeddings wide')


