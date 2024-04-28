import json
file_path ='/Users/Sophie/Documents/github/NLP_knowledgeEnhanced/data_biolama/ctd/entities/NCBI_human_gene_20210407.json'
f = open(file_path)
data = json.load(f)

print(data.keys())