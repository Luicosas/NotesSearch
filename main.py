from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import os
import logging
import sys
import pickle
from tqdm import tqdm


annoyDisType = 'dot'

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
    passages = []
    embeddings = []
    for filepath in tqdm(filepaths):
        try:
            content = open(filepath, 'r').read()
            # print(content)
        except:
            logging.error("can't open " + filepath)
            continue
        chunk_size = model.max_seq_length * 2
        for start in range(0, max(1, len(content) - chunk_size), int(chunk_size / 2)):
            # model will truncate automatically
            embedding = model.encode(content[start:]) # i hope python is smart and content[start:] does not copy string from start to end
            # paths.append(filepath)
            paths.append(filepath)
            passages.append(content[start : start + chunk_size])
            embeddings.append(embedding)
        # print_progress_part("Embedding", (idx + 1) / len(filepaths))
    return (paths, passages, embeddings)

# create and write annoy file
# annfilepath is the filepath to save to
def create_ann(embeddings, annfilepath):
    dim = len(embeddings[0])
    t = AnnoyIndex(dim, annoyDisType)
    for idx, embedding in enumerate(embeddings):
        t.add_item(idx, embedding)
    t.build(100)
    t.save(annfilepath)


# returns the index of the embedding of nearest neighbor
def query(query, model, annfilepath):
    query_embedding = model.encode(query)
    dim = len(query_embedding)
    u = AnnoyIndex(dim, annoyDisType)
    u.load(annfilepath)
    return u.get_nns_by_vector(query_embedding, 10)

def rank(query, matches, passages, cross_encoder_model):
    model_inputs = [[query, passages[id]] for id in matches]
    scores = cross_encoder_model.predict(model_inputs)
    return sorted(zip(scores, matches), reverse=True)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

    if(len(sys.argv) != 4):
        print("Usage: python3 main.py build (notes-dir) (data dir name), python3 main.py search \"(query string)\" (data dir name)")
        sys.exit(1)

    datadirname = os.path.join("./data", sys.argv[3])
    if not os.path.exists(datadirname):
        os.makedirs(datadirname)

    pathsfilepath = os.path.join(datadirname, "paths")
    passagesfilepath = os.path.join(datadirname, "passages")
    annfilepath = os.path.join(datadirname, "data.ann")
    embedding_model = SentenceTransformer("msmarco-distilbert-base-tas-b")
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")
    
    option = sys.argv[1]
    if option == "build":
            notes_dir = sys.argv[2]
            filepaths = get_filepaths(notes_dir)
            (paths, passages, embeddings) = make_embeddings(embedding_model, filepaths)
            pickle.dump(paths, open(pathsfilepath, 'wb'))
            print("paths file successfully pickle dumped at", pathsfilepath)
            pickle.dump(passages, open(passagesfilepath, 'wb'))
            print("passages file successfully pickle dumped at", passagesfilepath)
            create_ann(embeddings, annfilepath)
            print("ann file successfully built at", annfilepath)
    elif option == "search":
            query_string = sys.argv[2]
            query_ans = query(query_string, embedding_model, annfilepath)
            paths = pickle.load(open(pathsfilepath, 'rb'))
            passages = pickle.load(open(passagesfilepath, 'rb'))
            answers = rank(query_string, query_ans, passages, cross_encoder_model)
            for (idx, (score, id)) in enumerate(answers):
                if (idx < 3): 
                    print(paths[id], "has score: ", score)
                    print(passages[id])
                    print("-------------\n")
    else: 
        print("Option", option, "not found")
