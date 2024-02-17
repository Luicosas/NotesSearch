# Notes Search

The plan is to use the sentence transformers to create embeddings that we can later look up with the annoy nearest neighbor search to find relevant files.
We might need to implement some sort of caching embeddings with maybe sqlite later to speed up startup.  

# Offline usage
optionally make a model directory and in the model directory git clone the [embedding model](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b/tree/main) and [cross encoder model](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2/tree/main). 

The folder structure (tree -d) output should look like 
models/
├── msmarco-distilbert-base-tas-b
│   └── 1_Pooling
└── ms-marco-TinyBERT-L-2

Remember to "git-lfs pull" after git cloning to get the model files. The main.py automatically checks these two folders before trying to load the models from the internet

# Usage
Usage: 
```source venv/bin/activate```

First create the embeddings for the notes and the annoy tree with the following:
```python3 main.py build (notes-dir) (data directory name)```
ex. python3 main.py build ~/Notes notes

Do semantic search on the notes with the following:
```python3 main.py search "(query string) (data directory name)"```
ex. python3 main.py search "ssh" notes

# Useful references
[Annoy usage](https://github.com/spotify/annoy)

[Sentence transformer usage](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)

[ReRanking](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
