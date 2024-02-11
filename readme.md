# Notes Search

The plan is to use the sentence transformers to create embeddings that we can later look up with the annoy nearest neighbor search to find relevant files.
We might need to implement some sort of caching embeddings with maybe sqlite later to speed up startup.  

# Usage
Usage: 
```source venv/bin/activate```

First create the embeddings for the notes and the annoy tree with the following:
```python3 main.py build (notes-dir)```

Do semantic search on the notes with the following:
```python3 main.py search "(query string)"```

# Useful references
[Annoy usage](https://github.com/spotify/annoy)

[Sentence transformer usage](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)

[ReRanking](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
