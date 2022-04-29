from argparse import ArgumentParser
import random
import pandas as pd
import numpy as np
import csv
import pickle
import json
import tqdm
import argparse

from transformers import *
from constants import *
from external_knowledge.umls import umls_search_concepts, umls_extract_network
from data.biorelex import load_biorelex_dataset

# All file names
SRDEF = 'resources/SRDEF'
SRSTRE2 = 'resources/SRSTRE2'

def generate_embedding_pickle(text2graph, file_path='resources/embeddings.csv'):
    # Done reading dictionary from csv by chunk
    print('Loading csv dict...')
    chunksize=int(1e4); orig_dict = {}
    with pd.read_csv(file_path, chunksize=chunksize, header=None) as reader:
        for chunk in tqdm.tqdm(reader):
            rows = chunk.values
            for row in rows:
                orig_dict[str(row[0])] = np.array(row[1:], dtype=np.double)
    print('Done loading csv. Saving important info...')
    # Saving only important embedding
    new_dict = {}
    cnt = 0
    for g_info in tqdm.tqdm(text2graph.values()):
        if not type(g_info) is dict:
            continue
        nodes = list(set(g_info['nodes']))
        for n in nodes:
            if n in orig_dict:
                new_dict[n] = orig_dict[n]
            else:
                cnt += 1
    print(f'Cannot find {cnt} entities embedding')
    with open(UMLS_EMBS, 'wb') as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    print('Done saving info')
    
def find_until_meet(pattern, s, n):
    ind = s.find(pattern); n -= 1
    while n > 0:
        ind = s.find(pattern, ind+1)
        n -= 1
    return s[ind+1:]

# Generate UMLS_RELTYPES_FILE (UMLS_SEMTYPES_FILE not needed)
def generate_txt():
    with open(SRDEF, 'r') as f:
        with open(UMLS_RELTYPES_FILE, 'w') as rel_file:
            for line in f:
                if(line.startswith("RL")):
                    rel_file.write(find_until_meet('|', line, 7))

def generate_relation_dict():
    abbrev = {}
    with open(SRDEF, 'r') as f:
        for line in f:
            fields = line.split("|")
            abbrev[fields[2]] = fields[8]
    d = {}
    with open('resources/SRSTRE2', 'r') as f:
        for line in f:
            fields = line.split("|")
            d[(abbrev[fields[0]], abbrev[fields[2]])] = abbrev[fields[1]]
    
    with open('resources/relation.pickle', 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

def get(text, text2graph, relation_dict, embeddings=None):
    if text in text2graph:
        return
    # Use umls_search_concepts to return relevant concepts from text
    sents = [text]
    search_result, _ = umls_search_concepts(sents)
    search_result = search_result[0]
    concepts = search_result['concepts']
    # Add all cuis of concepts to nodes list
    nodes = []
    '''
    for concept in concepts:
        #print(concept)
        if embeddings is None or concept['cui'] in embeddings:
            nodes.append(concept['cui'])
    '''
    # Edge list include cuis of two concepts, and the (abbreviated) relation that joins them
    edges = []; nodes_set = set()
    for concept_start in concepts:
        for concept_end in concepts:
            for s_start in concept_start['semtypes']:
                for s_end in concept_end['semtypes']:
                    check_embed_start = embeddings is None or concept_start['cui'] in embeddings
                    check_embed_end = embeddings is None or concept_end['cui'] in embeddings
                    if (s_start, s_end) in relation_dict and check_embed_start and check_embed_end:
                        edges.append((concept_start['cui'], 
                            relation_dict[(s_start, s_end)], concept_end['cui']))
                        nodes_set.add(concept_start['cui'])
                        nodes_set.add(concept_end['cui'])
    # Node list from edge list:
    nodes = list(nodes_set)

    # Store final info that include nodes and edges
    text2graph[text] = {'nodes': nodes, 'edges': edges}
    
def get_multiple_texts(texts, relation_dict, embeddings=None):
    # Load relation dictionary
    with open('resources/relation.pickle', 'rb') as f:
        relation_dict = pickle.load(f)
    text2graph = {}
    # get each text in array of texts if not saved before
    for text in tqdm.tqdm(texts):
        get(text, text2graph, relation_dict, embeddings)
    # Save dictionary to text2graph.pkl
    with open('resources/text2graph.pkl', 'wb') as f:
        pickle.dump(text2graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    return text2graph

# Read raw instances
def read_texts(file_path, mode='biorelex'):
    texts = []
    if mode == 'biorelex':
        raw_insts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_insts += json.load(f)
        # Construct data_insts
        for inst in raw_insts:
            texts.append(inst['text']) 
        return texts

    if mode == 'ade':
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        for raw_inst in data:
            tokens = raw_inst['tokens']
            texts.append(' '.join(tokens))
        return texts
    
def print_graph_ex(text2graph, text):
    g_info = text2graph[text]
    nodes, edges = list(set(g_info['nodes'])), list(set(g_info['edges'])) # Nodes is a list of CUID
    print('Nodes:', nodes)
    print ('Edges:')
    for n1, edge_type, n2 in edges:
        print(n1, n2, edge_type)

def get_important_files(texts):
    # Load relation dict (relation.pickle)
    with open('resources/relation.pickle', 'rb') as f:
        relation_dict = pickle.load(f)
    # General initial text2graph
    get_multiple_texts(texts, relation_dict)
    # Print one example
    with open('resources/text2graph.pkl', 'rb') as f:
        text2graph=pickle.load(f)
    print_graph_ex(text2graph, texts[1])
    print('Done text2graph without embedding')
    # Generate relevant embedding
    with open('resources/text2graph.pkl', 'rb') as f:
        text2graph=pickle.load(f)
    generate_embedding_pickle(text2graph)
    print('Done generating relevant embedding')
    # Generate final text2graph
    with open(UMLS_EMBS, 'rb') as f:
        embeddings = pickle.load(f)
    with open('resources/relation.pickle', 'rb') as f:
        relation_dict = pickle.load(f) # Load relation dict (relation.pickle)
    get_multiple_texts(texts, relation_dict, embeddings)
    print('Done text2graph with embedding')
    # Test if umls_extract_network works well
    with open('resources/text2graph.pkl', 'rb') as f:
        text2graph=pickle.load(f)
        for text in texts:
            umls_extract_network(text)

def get_texts(text_mode, max_train=0, max_dev=0):
    # Set paths to data
    if text_mode == 'biorelex':
        train_path, dev_path = 'resources/biorelex/train.json', 'resources/biorelex/dev.json'
    if text_mode == 'ade':
        train_path, dev_path = 'resources/ade/ade_split_0_test.json', 'resources/ade/ade_split_0_train.json'
    # Get texts
    train_texts = read_texts(train_path, text_mode) 
    if max_train != 0:
        random.Random(SEED).shuffle(train_texts)
        train_texts = train_texts[:max_train]
    dev_texts = read_texts(dev_path, text_mode)
    if max_dev != 0:
        random.Random(SEED).shuffle(dev_texts)
        dev_texts = dev_texts[:max_dev]
    print('Text read.')
    return train_texts + dev_texts

def word_to_cui(w):
    sents = [w]
    search_result, _ = umls_search_concepts(sents)
    concepts = search_result[0]['concepts']
    return concepts[0]['cui']
    
def adjust_common_embeddings():
    # Open uuid common embedding file
    with open(COMMON_EMBS_FILE_UUID, 'rb') as f:
        uuid_common_embs = pickle.load(f)
    # Go over this file and change word keys to cuid keys
    common_embs = {}
    for w, emb_vec in uuid_common_embs.items():
        common_embs[word_to_cui(w)] = emb_vec
    # Save final dictionary to the common embedding file
    with open(COMMON_EMBS_FILE, 'wb') as f:
        pickle.dump(common_embs, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating files')
    parser.add_argument('--text_mode', type=str, default='None', help='Which source to read text from')
    parser.add_argument('--gen_rel', type=bool, default=False, help='Whether to generate relation.pickle')
    parser.add_argument('--gen_files', type=bool, default=False, help='Whether to generate important files')
    parser.add_argument('--max_train', type=int, default=0, help='Maximum training examples allowed')
    parser.add_argument('--max_dev', type=int, default=0, help='Maximum training examples allowed')
    args = parser.parse_args()
    # Generate relation dictionary
    if args.gen_rel:
        generate_relation_dict() 
    # Generate important files: text2graph and umls_embs
    if args.gen_files:
        # Read texts
        texts = get_texts(args.text_mode, args.max_train, args.max_dev)
        get_important_files(texts)
