TASK=wikidata 
PROMPT_PATH=BioLAMA1/data/${TASK}/prompts/manual.jsonl
TEST_PATH=BioLAMA1/data/${TASK}/triples_processed/*/testcopy.jsonl

python3 BioLAMA1/BioLAMA/run_manual.py \
    --prompt_path ${PROMPT_PATH} \
    --test_path "${TEST_PATH}" \
    --init_method confidence \
    --iter_method none \
    --num_mask 10 \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --pid 'P780'\
    --output_dir data_testing