# An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs
This is the official repo for the paper "An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs".  

🎉 [News]: [2026.04] Our paper is accepted by ACL 2026 Findings.

## Download the Datsets
Download TREC-DL test data from [TREC-DL19 and TREC-DL20](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html);
WebAP from [WebAP](https://ciir.cs.umass.edu/downloads/WebAP/index.html); 
NQ from [NQ](https://ai.google.com/research/NaturalQuestions/download);
GTI-NQ from [GTI-NQ](https://drive.google.com/drive/folders/1zmj2QiAxqsNfDf7iihYYKsdhL-WbvAYb?usp=drive_link)

## Get the retrieval data for WebAP and TREC DL
BM25 for the TREC and WebAP datasets, RocketQAv2 for the NQ dataset.

## Create the candidate passage list for WebAP and TREC DL
We have also provided the final datasetsets of WebAP and TREC DL in the "data/" folder.


## ITEM via LLMs
Taking the testing of Mistral-7B as an example, we demonstrated the use of four methods: ITEM-As, ITEM-ARs, and single-shot-utility-judgments
```
python mistral/item-As-ImpA.py

```


