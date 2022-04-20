import pandas as pd
import csv
import pickle
import json
import tqdm

from transformers import *
from constants import *
from external_knowledge.umls import umls_search_concepts
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
                orig_dict[str(row[0])] = row[1:]
    print('Done loading csv. Saving important info...')
    # Saving only important embedding
    new_dict = {}
    for g_info in tqdm.tqdm(text2graph.values()):
        if not type(g_info) is dict:
            continue
        nodes = list(set(g_info['nodes']))
        for n in nodes:
            if n in orig_dict:
                new_dict[n] = orig_dict[n]
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
    with open(SRDEF, 'r') as r_def_f:
        for line in r_def_f:
            fields = line.split("|")
            abbrev[fields[2]] = fields[8]
    d = {}
    with open('resources/SRSTRE2', 'r') as r_f:
        for line in r_f:
            fields = line.split("|")
            d[(abbrev[fields[0]], abbrev[fields[2]])] = abbrev[fields[1]]
    
    with open('resources/relation.pickle', 'wb') as w_f:
        pickle.dump(d, w_f, protocol=pickle.HIGHEST_PROTOCOL)

def get(text, text2graph, relation_dict):
    if text in text2graph:
        return
    # Use umls_search_concepts to return relevant concepts from text
    sents = [text]
    search_result, _ = umls_search_concepts(sents)
    search_result = search_result[0]
    concepts = search_result['concepts']
    # Add all cuis of concepts to nodes list
    nodes = []
    for concept in concepts:
        #print(concept)
        nodes.append(concept['cui'])
    # Edge list include cuis of two concepts, and the (abbreviated) relation that joins them
    edges = []
    for concept_start in concepts:
        for concept_end in concepts:
            for s_start in concept_start['semtypes']:
                for s_end in concept_end['semtypes']:
                    if (s_start, s_end) in relation_dict:
                        edges.append((concept_start['cui'], 
                            relation_dict[(s_start, s_end)], concept_end['cui']))
    # Store final info that include nodes and edges
    text2graph[text] = {'nodes': nodes, 'edges': edges}
    
def get_multiple_texts(texts, relation_dict):
    # Load relation dictionary
    with open('resources/relation.pickle', 'rb') as rel_dict_f:
        relation_dict = pickle.load(rel_dict_f)
    # Load dictionary from text2graph.pkl
    with open('resources/text2graph.pkl', 'rb') as r_f:
        text2graph = pickle.load(r_f)
    # get each text in array of texts if not saved before
    for text in tqdm.tqdm(texts):
        get(text, text2graph, relation_dict)
    # Save dictionary to text2graph.pkl
    with open('resources/text2graph.pkl', 'wb') as w_f:
        pickle.dump(text2graph, w_f, protocol=pickle.HIGHEST_PROTOCOL)

    return text2graph

# Read raw instances
def read_biorelex_texts(file_path):
    raw_insts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_insts += json.load(f)

    # Construct data_insts
    texts = []
    for inst in raw_insts:
        texts.append(inst['text'])
    
    return texts
        

if __name__ == '__main__':
    # Get texts
    texts = None
    text_mode = 'biorelex' #'None' #
    if text_mode == 'biorelex':
        train_path, dev_path = 'resources/biorelex/train.json', 'resources/biorelex/dev.json'
        texts = read_biorelex_texts(train_path) + read_biorelex_texts(dev_path)
        first_5_texts = texts[:5]
        #print(first_5_texts)
    ## Update text2graph.pkl
    update_text_to_graph = False; gen_rel=False
    if update_text_to_graph:
        # Generate (and load) relation dictionary
        if gen_rel:
            generate_relation_dict() # Generate relation dict
        with open('resources/relation.pickle', 'rb') as r_f:
            relation_dict = pickle.load(r_f)
        get_multiple_texts(texts, relation_dict)
        # Print one example
        with open('resources/text2graph.pkl', 'rb') as r_f:
            text2graph=pickle.load(r_f)
        g_info = text2graph[first_5_texts[1]]
        nodes, edges = list(set(g_info['nodes'])), list(set(g_info['edges'])) # Nodes is a list of CUID
        print('Nodes:', nodes)
        print ('Edges:')
        for n1, edge_type, n2 in edges:
            print(n1, n2, edge_type)

    pickle_embedding = True
    if pickle_embedding:
        with open('resources/text2graph.pkl', 'rb') as r_f:
            text2graph=pickle.load(r_f)
        # Generate embedding files
        generate_embedding_pickle(text2graph)
        # Load a few nodes from some text of biorelex and see their embeddings
        with open(UMLS_EMBS, 'rb') as emb_f:
            umls_embs =pickle.load(emb_f)
            print('Length dict: ', len(umls_embs))
            g_info = text2graph[first_5_texts[2]]
            nodes, edges = list(set(g_info['nodes'])), list(set(g_info['edges'])) # Nodes is a list of CUID
            for n in nodes:
                print(umls_embs[n])