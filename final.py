import sys
import pickle
import os

from LoadDataset import load_dataset
from GenerateEmbeddings import generate_embeddings_narrow, generate_embeddings_wide

# TODO: Change this later

# Initialize two directories
input_dir= sys.argv[1]
output_dir= sys.argv[2]


# Read the two datasets
input_path_narrow= input_dir + '/dataset-narrow'
dataset_narrow= load_dataset(input_path_narrow)

input_path_wide= input_dir + '/dataset-wide'
dataset_wide= load_dataset(input_path_wide)

# Create two subfolders within output dir, dataset-narrow and dataset-wide
output_path_narrow= output_dir + '/dataset-narrow'
output_path_wide= output_dir + '/dataset-wide'

os.mkdir(output_path_narrow)
os.mkdir(output_path_wide)


# print(dataset_narrow[1060:1070][1])
# print(dataset_wide[1060:1070][1])
# Generate Embeddings and predict
generate_embeddings_narrow(dataset_narrow, output_path_narrow)
generate_embeddings_wide(dataset_wide, output_path_wide)

