# %%
import argparse
import json
import os 
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--prompt_path", default='./data/wikidata/prompts/manual.jsonl')
parser.add_argument("-t","--test_path", required=True)
parser.add_argument("--pid", default=None)
args = parser.parse_args()



def test_path(prompt_path,test_p, pids):
    result = [[2,3],[3,4]]
    cnt = 0
    files  = glob(test_p)
    print(prompt_path)
    print(test_p)
    print(files)
    print(pids)
    if pids: # e.g., P1050
        new_files = []
        pids = list(dict.fromkeys(pids.split(",")))
        for file in files:
            if file.split("/")[-2] in pids:
                new_files.append(file)
        files = new_files
    print(files)
        

if __name__ == '__main__':
    test_path(args.prompt_path,args.test_path,args.pid)


# %%
'''
python3 BioLAMA/BioLAMA/run_manual.py \
    --prompt_path ${PROMPT_PATH} \
    --test_path "${TEST_PATH}" \
    --init_method confidence \
    --iter_method none \
    --num_mask 10 \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --pid 'CD1'\
    --output_dir sumo/data_testing
'''