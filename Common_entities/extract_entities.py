
# %%
from curses import raw
from collections import defaultdict
import os
import glob
import numpy as np
import json


def read_sub(sources):
    common_entities = defaultdict(list)
    cnt = 0
    for source_file in sources:
        files_path = glob.glob(os.path.join(source_file, '**/*.jsonl'),recursive=True)
        for idx,input_file in enumerate(files_path):
            with open(input_file) as f:
                for line in f:
                    sample = json.loads(line)
                    sub_label = sample['sub_label']
                    if sub_label not in common_entities:
                        cnt += 1
                        common_entities[sub_label].append(sample)
                    else: 
                        common_entities[sub_label].append(sample)
    print(f'Data has {cnt} entities, common list has {len(common_entities)} entities')
    return common_entities

def filter_common(common_entites, thres = 3):
    remove = []
    for sub,rel_list in common_entites.items():
        if len(rel_list) == 1:
            if len(rel_list[0]["obj_labels"]) <=3:
                remove.append(sub)
    for ent in remove: 
        common_entites.pop(ent)

    print(f'List now has {len(common_entites)}')
    return common_entites



from helpers import *
def process_BioRelEx(file_path):
    # Read raw instances
    BRLEX_entities = []
    raw_insts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_insts += json.load(f)
    
    for inst in raw_insts:
        id, text = inst['id'], inst['text']
        raw_entites = inst.get('entities', [])
        for eid, e in enumerate(raw_entites):
            # print(e,'\n')
            entity_mentions = []
            for name in e['names']:
                BRLEX_entities.append(name)
    return BRLEX_entities

def checking_obj(common_entities):
    # Can use sub_uris vs obj_ris?
    cnt_by_sub = []
    sub_list = []
    obj_count = {}
    
    for sub in common_entities.keys():
        if len(common_entities[sub]) >0:
            sub_list.append(common_entities[sub][0]["sub_uri"])
    print('finish Getting uri list from subject \n')
    print('Counting obj in entities list')
    for key, rels in common_entities.items():
        obj_per_sub = set()
        belong = 0
        for rel in rels:
            for obj_uris in rel['obj_uris']:
                obj_per_sub.add(obj_uris)
        for obj in obj_per_sub:
            if obj in sub_list:
                belong +=1
        if belong >0:
            obj_count[key] = (len(obj_per_sub),belong,common_entities[key][0]["sub_aliases"])
    return obj_count
        



# %%
