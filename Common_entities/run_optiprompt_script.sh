# Set TASK to 'ctd' for CTD or 'umls' for UMLS
# Set MODEL to 'bert-base-cased' for BERT or 'dmis-lab/biobert-base-cased-v1.2' for BioBERT
TASK=wikidata
MODEL=dmis-lab/biobert-base-cased-v1.2
PROMPT_PATH=BioLAMA/data/${TASK}/prompts/manual.jsonl
TRAIN_PATH=./data/${TASK}/triples_processed/*/train.jsonl
DEV_PATH=./data/${TASK}/triples_processed/*/dev.jsonl
TEST_PATH=BioLAMA/data/${TASK}/triples_processed/*/testcopy.jsonl

python ./BioLAMA/run_optiprompt.py \
    --model_name_or_path ${MODEL} \
    --train_path "${TRAIN_PATH}" \
    --dev_path "${DEV_PATH}" \
    --test_path "${TEST_PATH}" \
    --prompt_path ${PROMPT_PATH} \
    --num_mask 10 \
    --init_method confidence \
    --iter_method none \
    --max_iter 10 \
    --beam_size 5 \
    --batch_size 16 \
    --lr 3e-3 \
    --epochs 10 \
    --seed 0 \
    --prompt_token_len 5 \
    --init_manual_template \
    --pid 'P780'\
    --output_dir sumo/data_testing/${TASK}_optiprompt