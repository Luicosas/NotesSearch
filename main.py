from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, util
import os
import logging
import sys

# gets the files of in a directory
def get_filepaths(directory):
    ignore_list = ['.git']
    files_list =[]
    for root, dirs, files in os.walk(directory):
        if any(map(lambda banned : root.find(banned) != -1, ignore_list)):
            continue
        # files[:] = [f for f in files if f not in ignore_list]
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list

# create embeddings
def make_embeddings(model, filepaths):
    paths = []
    embeddings = []
    for filepath in filepaths:
        try:
            content = open(filepath, 'r').read()
            print(content)
        except:
            logging.error("can't open " + filepath)
            continue
        embedding = model.encode(content)
        paths.append(filepath)
        embeddings.append(embedding) 
    return (paths, embeddings)

# create and write annoy file
# annfilepath is the filepath to save to
def create_ann(embeddings, annfilepath):
    dim = len(embeddings[0])
    t = AnnoyIndex(dim, 'angular')
    for idx, embedding in enumerate(embeddings):
        t.add_item(idx, embedding)
    t.build(100)
    t.save(annfilepath)


# returns the index of the embedding of nearest neighbor
def query(query, model, annfilepath):
    query_embedding = model.encode(query)
    dim = len(query_embedding)
    u = AnnoyIndex(dim, 'angular')
    u.load(annfilepath)
    return u.get_nns_by_vector(query_embedding, 10)


if __name__ == "__main__":
    notes_dir = sys.argv[1]

    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
    
    model = SentenceTransformer("msmarco-distilroberta-base-v3")

    (paths, embeddings) = make_embeddings(model, notes_dir)

    query_embedding = model.encode("How big is London")
    passage_embedding = model.encode("London has 9,787,426 inhabitants at the 2011 census")

    print("Similarity:", util.cos_sim(query_embedding, passage_embedding))

    f = 40  # Length of item vector that will be indexed

    t = AnnoyIndex(f, 'angular')
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)

    t.build(10) # 10 trees
    t.save('test.ann')

    # ...

    u = AnnoyIndex(f, 'angular')
    u.load('test.ann') # super fast, will just mmap the file
    print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors