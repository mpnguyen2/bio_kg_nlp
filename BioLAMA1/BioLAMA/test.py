# %%
import torch
import transformers
import argparse
import os

#!/opt/anaconda3/envs/bio_nlp/bin python3

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--text", required=True)
    parser.add_argument("--model", default='haha')
    
    args = parser.parse_args()
    A = 3
    print(f'model is {args.model}')


if __name__ == '__main__':
    main()

# %%
