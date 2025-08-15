import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tokenizers import Tokenizer
from seq2tensor import s2t

def create_tokenizer_custom(file):

    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

def load_and_preprocess_data_tokenizer(positive_path, negative_path, max_length,tokenizer):

    print("Loading data...")
    positive_df = pd.read_excel(positive_path)
    negative_df = pd.read_excel(negative_path)
    
    positive_df['label'] = '1'
    negative_df['label'] = '0'
    all_data = pd.concat([positive_df, negative_df], ignore_index=True)

    protein_to_id = {}
    current_id = 0
    seq_array = []  
    protein_length = []
    raw_data = []  


    pad_token_id = tokenizer.encode('<|pad|>').ids[0]

    for idx, row in all_data.iterrows():
        prot1_seq = row['SequenceA']
        prot2_seq = row['SequenceB']
        label = row['label']

        prot1_seq = "1" + prot1_seq + "2"
        prot2_seq = "1" + prot2_seq + "2"

        encodedA = tokenizer.encode(prot1_seq).ids
        encodedB = tokenizer.encode(prot2_seq).ids

        if len(encodedA) > max_length:
            encodedA = encodedA[:max_length]
            attention_maskA = [1] * max_length
        else:
            encodedA = encodedA + [pad_token_id] * (max_length - len(encodedA))
            attention_maskA = [1 if id != pad_token_id else 0 for id in encodedA]

        if len(encodedB) > max_length:
            encodedB = encodedB[:max_length]
            attention_maskB = [1] * max_length
        else:
            encodedB = encodedB + [pad_token_id] * (max_length - len(encodedB))
            attention_maskB = [1 if id != pad_token_id else 0 for id in encodedB]

        if prot1_seq not in protein_to_id:
            protein_to_id[prot1_seq] = current_id
            seq_array.append(encodedA)
            protein_length.append(len(encodedA))
            current_id += 1
        id1 = protein_to_id[prot1_seq]

        if prot2_seq not in protein_to_id:
            protein_to_id[prot2_seq] = current_id
            seq_array.append(encodedB)
            protein_length.append(len(encodedB))
            current_id += 1
        id2 = protein_to_id[prot2_seq]

        raw_data.append([id1, id2, label])
    seq_array = np.array(seq_array)
    print(f"Total pairs: {len(raw_data)}")
    
    seq_index1 = np.array([line[0] for line in raw_data])
    seq_index2 = np.array([line[1] for line in raw_data])
    
    class_map = {'0': 0, '1': 1}
    class_labels = np.zeros((len(raw_data), 2))
    for i in range(len(raw_data)):
        class_labels[i][class_map[raw_data[i][2]]] = 1.0
    
    return seq_array, seq_index1, seq_index2, class_labels

def load_and_preprocess_data_seq2t(positive_path, negative_path, max_length,seq2t):

    print("Loading data...")
    positive_df = pd.read_excel(positive_path)
    negative_df = pd.read_excel(negative_path)#random_pairs_with_sequences,3_2neg_protein_pairs_sequence

    positive_df['label'] = '1'
    negative_df['label'] = '0'
    all_data = pd.concat([positive_df, negative_df], ignore_index=True)

    protein_to_id = {}
    current_id = 0
    seq_array = []
    protein_length=[]
    raw_data = []
    for idx, row in all_data.iterrows():
        prot1_seq = row['SequenceA']
        prot2_seq = row['SequenceB']
        label = row['label']

        if prot1_seq not in protein_to_id:
            protein_to_id[prot1_seq] = current_id
            seq_array.append(prot1_seq)
            protein_length.append(len(prot1_seq))
            current_id += 1
        id1 = protein_to_id[prot1_seq]

        if prot2_seq not in protein_to_id:
            protein_to_id[prot2_seq] = current_id
            seq_array.append(prot2_seq)
            protein_length.append(len(prot2_seq))
            current_id += 1
        id2 = protein_to_id[prot2_seq]

        raw_data.append([id1, id2, label])

    print (len(raw_data))

    seq_tensor = np.array([seq2t.embed_normalized(line, max_length) for line in tqdm(seq_array)])

    sid1_index = 0
    sid2_index = 1
    seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
    seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

    class_map = {'0':0,'1':1}
    class_labels = np.zeros((len(raw_data), 2))
    for i in range(len(raw_data)):
        class_labels[i][class_map[raw_data[i][-1]]] = 1.
    
    return seq_tensor, seq_index1, seq_index2, class_labels