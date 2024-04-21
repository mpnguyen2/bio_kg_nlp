import argparse, pickle, tqdm, random, json
import numpy as np
import matplotlib.pyplot as plt
from constants import *

## TMP TextReader
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

# distribution of ratio # common nodes/ total nodes
# distribution of ration # total degree of common nodes / total degree of total nodes
# node_dict keeps track of how many sentences each node appear in
def graph_cnt(text, text2graph, common_embs, sentence_cnt, embed_nodes):
    g_info = text2graph[text]
    # Nodes is a list of CUID
    nodes, edges = list(set(g_info['nodes'])), list(set(g_info['edges']))
    cnt_node = 0; cnt_edge = 0
    # Cnt node
    for n in nodes:
        if n in sentence_cnt:
            sentence_cnt[n] += 1
        else:
            sentence_cnt[n] = 1
        if n in common_embs:
            cnt_node += 1
            embed_nodes.add(n)
            
    for n1, edge_type, n2 in edges:
        if n1 in common_embs and n2 in common_embs:
            cnt_edge += 1

    return cnt_node, len(nodes), cnt_edge, len(edges)

def graph_cnt_all(texts, text2graph, common_embs, num_bins=20, name='wiki'):
    # Collect stat over all texts
    print('Getting stats...')
    sentence_cnt = {}; embed_nodes = set()
    node_ratios = []; edge_ratios = []
    total_node_emb = 0; total_node = 0; total_edge_emb = 0; total_edge = 0
    for text in tqdm.tqdm(texts, leave=False):
        cnt_node_emb, cnt_node, cnt_edge_emb, cnt_edge = graph_cnt(text, text2graph, common_embs, sentence_cnt, embed_nodes)
        total_node_emb += cnt_node_emb; total_node += cnt_node
        total_edge_emb += cnt_edge_emb; total_edge += cnt_edge
        if cnt_node_emb != 0 and cnt_edge_emb/cnt_edge >= 0.01:
            node_ratios.append(cnt_node_emb/cnt_node)
            edge_ratios.append(cnt_edge_emb/cnt_edge)
    # Print total stat
    print('\nDistinct stats: we count only distinct nodes over all sentences')
    print('Total distinct nodes: ', len(sentence_cnt))
    print('Total distinct common nodes: ', len(embed_nodes))
    print('\nFrequent nodes stat:')
    cnt1 = 0; cnt2 = 0
    for n in sentence_cnt:
        if sentence_cnt[n] >= 10:
            cnt1 += 1
            if n in embed_nodes:
                cnt2 += 1
    print('Total nodes appear in >= 10 sentence: ', cnt1)
    print('Total nodes in common with appear in >= 10 sentence: ', cnt2)
    print('Percentage: ', cnt2/cnt1)
    # nodes and edges summing over all sentences
    print('\nRepeated stats: we now count repeated nodes (and edges) over all sentences')
    print('Node: {}, Common node: {}'.format(total_node, total_node_emb))
    print('Percentage of nodes saved: ', total_node_emb/total_node)
    print('Edge: {}, Common edge: {}'.format(total_edge, total_edge_emb))
    print('Percentage of edges saved: ', total_edge_emb/total_edge)
    # Plot histogram 
    weights=np.ones(len(node_ratios))/len(texts)
    # Node ratio distribution
    plt.title('Distribution of node ratios of embedded vs total')
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel('Ratio')
    plt.ylabel('Percentage of sentences')
    plt.hist(node_ratios, weights=weights, bins=num_bins)
    plt.savefig('out/' + name + '_node_ratios.jpg')
    # Edge ratio distribution
    plt.clf()
    plt.title('Distribution of edge ratios of embedded vs total')
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel('Ratio')
    plt.ylabel('Percentage of sentences')
    plt.hist(edge_ratios, weights=weights, bins=num_bins)
    plt.savefig('out/' + name + '_edge_ratios.jpg')
    # Number of sentence ratio per node
    sentence_ratios = [sentence_cnt[n] for n in embed_nodes]
    plt.clf()
    plt.title('Distribution of sentence ratios of nodes in embedded')
    #plt.xticks(np.arange(0, 1, step=0.01))
    plt.xlabel('Ratio')
    plt.ylabel('Number of words')
    plt.hist(sentence_ratios, bins=num_bins)
    plt.savefig('out/' + name + '_sentence_ratios.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating files')
    parser.add_argument('--mode', type=str, default='None', help='Which source to read text from')
    parser.add_argument('--name', type=str, default='wiki', help='Embedding name')
    parser.add_argument('--num_bins', type=int, default=100, help='Number of bins for distribution display')
    args = parser.parse_args()
    # Load text
    text_reader = TextReader(args.mode)
    texts = text_reader.get_texts()
    # Load embedding
    with open('resources/' + args.name + '_common_embs.pkl', 'rb') as f:
        common_embs = pickle.load(f)
    print('Common embedding length: ', len(common_embs))
    # Get stats
    text2graph = pickle.load(open(UMLS_TEXT2GRAPH_FILE, 'rb'))
    graph_cnt_all(texts, text2graph, common_embs, args.num_bins, args.name)
