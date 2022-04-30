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
from external_knowledge.umls import TEXT2GRAPH, umls_search_concepts, umls_extract_network
from data.biorelex import load_biorelex_dataset

#### Generating relation files ####
def find_until_meet(pattern, s, n):
    ind = s.find(pattern); n -= 1
    while n > 0:
        ind = s.find(pattern, ind+1)
        n -= 1
    return s[ind+1:]
# Generate UMLS_RELTYPES_FILE (UMLS_SEMTYPES_FILE not needed)
def generate_umls_reltypes():
    with open(SRDEF, 'r') as f:
        with open(UMLS_RELTYPES_FILE, 'w') as rel_file:
            for line in f:
                if(line.startswith("RL")):
                    rel_file.write(find_until_meet('|', line, 7))
# Generate RELATION_FILE
def generate_relation_dict():
    abbrev = {}
    with open(SRDEF, 'r') as f:
        for line in f:
            fields = line.split("|")
            abbrev[fields[2]] = fields[8]
    d = {}
    with open(SRSTRE2, 'r') as f:
        for line in f:
            fields = line.split("|")
            d[(abbrev[fields[0]], abbrev[fields[2]])] = abbrev[fields[1]]
    
    with open(RELATION_FILE, 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

#### Generating necessary text2graph and embedding files ####
# Helper functions
def get(text, text2graph, relation_dict, embeddings=None):
    if text in text2graph:
        return
    # Use umls_search_concepts to return relevant concepts from text
    sents = [text]
    search_result, _ = umls_search_concepts(sents)
    search_result = search_result[0]
    concepts = search_result['concepts']
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
    text2graph = {}
    # get each text in array of texts if not saved before
    for text in tqdm.tqdm(texts):
        get(text, text2graph, relation_dict, embeddings)
    return text2graph

# Print example of graph
def print_graph_ex(text2graph, text):
    g_info = text2graph[text]
    nodes, edges = list(set(g_info['nodes'])), list(set(g_info['edges'])) # Nodes is a list of CUID
    print('Nodes:', nodes)
    print ('Edges:')
    for n1, edge_type, n2 in edges:
        print(n1, n2, edge_type)

class FileGenerator:
    def __init__(self):
        # Map a pair of entity (abbreviated) types to a (abbreviated) relation
        self.relation_dict = {}
        # Map a CUID to corresponding Maldonado et al. embedding
        self.embed_dict = {}
        # Map text to a pair of dgl graph and nodes
        self.text2graph = {}
    
    def load_relation_dict(self):
        with open(RELATION_FILE, 'rb') as f:
            self.relation_dict = pickle.load(f)

    def generate_embedding(self, text2graph):
        # Done reading dictionary from csv by chunk
        print('Loading csv dict...')
        chunksize=int(1e4); orig_dict = {}
        with pd.read_csv(EMBED_DATABASE, chunksize=chunksize, header=None) as reader:
            for chunk in tqdm.tqdm(reader):
                rows = chunk.values
                for row in rows:
                    orig_dict[str(row[0])] = np.array(row[1:], dtype=np.double)
        print('Done loading csv. Saving important info...')
        # Saving only important embedding
        cnt = 0
        for g_info in tqdm.tqdm(text2graph.values()):
            if not type(g_info) is dict:
                continue
            nodes = list(set(g_info['nodes']))
            for n in nodes:
                if n in orig_dict:
                    self.embed_dict[n] = orig_dict[n]
                else:
                    cnt += 1
        print(f'Cannot find {cnt} entities embedding')
    
    def save_embedding(self):
        with open(UMLS_EMBS, 'wb') as f:
            pickle.dump(self.embed_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_text2graph(self):
        with open(UMLS_TEXT2GRAPH_FILE, 'wb') as f:
            pickle.dump(self.text2graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_files(self, texts):
         # Load relation dict (relation.pickle)
        self.load_relation_dict()
        # General initial text2graph
        print('Loaded relation dict. Generate text2graph without embedding...')
        self.text2graph = get_multiple_texts(texts, self.relation_dict)
        self.save_text2graph()
        print('Done text2graph without embedding. Generate relevant embedding...')
        # Generate relevant embedding
        self.generate_embedding(self.text2graph)
        self.save_embedding()
        print('Done generating relevant embedding. Generate text2graph with embedding...')
        # Generate final text2graph
        self.text2graph = get_multiple_texts(texts, self.relation_dict, self.embed_dict)
        print('Done text2graph with embedding')
        self.save_text2graph()
        
    def sanity_check(self, texts):
        for text in texts:
            umls_extract_network(text)
    
#### Text reader ####
class TextReader:
    def __init__(self, mode='biorelex', max_train=0, max_dev=0):
        self.mode = mode
        self.max_train = max_train
        self.max_dev = max_dev
        # paths to data
        if self.mode == 'biorelex':
            self.train_path = 'resources/biorelex/train.json'
            self.dev_path  = 'resources/biorelex/dev.json'
        if self.mode == 'ade':
            self.train_path = 'resources/ade/ade_split_0_train.json'
            self.dev_path = 'resources/ade/ade_split_0_test.json'

    def read_texts(self, file_path):
        texts = []
        if self.mode == 'biorelex':
            raw_insts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_insts += json.load(f)
            # Construct data_insts
            for inst in raw_insts:
                texts.append(inst['text']) 
            return texts

        if self.mode == 'ade':
            with open(file_path, 'r') as f:
                data = json.loads(f.read())
            for raw_inst in data:
                tokens = raw_inst['tokens']
                texts.append(' '.join(tokens))
            return texts
   
    def get_texts(self):
        print('Reading text...')
        # Get train text
        train_texts = self.read_texts(self.train_path) 
        if self.max_train != 0:
            random.Random(SEED).shuffle(train_texts)
            train_texts = train_texts[:self.max_train]
        # Get dev texts
        dev_texts = self.read_texts(self.dev_path)
        if self.max_dev != 0:
            random.Random(SEED).shuffle(dev_texts)
            dev_texts = dev_texts[:self.max_dev]
        print('Text read.')
        return train_texts + dev_texts

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
        generate_umls_reltypes()
        generate_relation_dict()
    # Generate important files: text2graph and umls_embs
    if args.gen_files:
        # Read texts
        text_reader = TextReader(args.text_mode, args.max_train, args.max_dev)
        texts = text_reader.get_texts()
        # Generate files
        file_generator = FileGenerator()
        #file_generator.generate_files(texts)
        # Sanity check
        file_generator.sanity_check(texts)