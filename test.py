from re import search
from external_knowledge.umls import umls_search_concepts

sent_test = "The effect was specific to rapamycin, as FK506, an immunosuppressant, that also binds FKBP12 but that does not target mTOR, had no effect on the interaction"
sents = [sent_test]
search_result, _ = umls_search_concepts(sents)
search_result = search_result[0]
for concept in search_result['concepts']:
    print(concept)
