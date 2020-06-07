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
input_path_wide= input_dir + '/dataset-wide'

# Create two subfolders within output dir, dataset-narrow and dataset-wide
output_path_narrow= output_dir + '/dataset-narrow'
output_path_wide= output_dir + '/dataset-wide'

os.mkdir(output_path_narrow)
# os.mkdir(output_path_wide)

dataset_narrow= load_dataset(input_path_narrow)
generate_embeddings_narrow(dataset_narrow[10:15], output_path_narrow)
del dataset_narrow

# dataset_wide= load_dataset(input_path_wide)
# generate_embeddings_wide(dataset_wide[:2], output_path_wide)
# del dataset_wide
