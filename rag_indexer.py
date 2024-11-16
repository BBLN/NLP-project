#from arxiv_embedding import embed_metadata, EMBEDDING_DIM, embed_parallel
from pinecone_db import get_pinecone_client, get_or_create_index, METRIC_L2
from datetime import datetime
from semanticscholar_wrapper import SemanticScholarWrapper
import datasets
from specter_embedding import Specter2Document
import concurrent
from tqdm import tqdm

def get_record(embedding, metadata):
    filtered_metadata_columns = ["id", "categories", "title"]
    metadata = {k: v for k, v in metadata.items() if k in filtered_metadata_columns}
    return {
        "id": f"{metadata['id']}#arxiv-metadata#scibert",
        "values": embedding,
        "metadata": metadata
    }

def build_arxiv_index():
    import arxiv_embedding
    pc = get_pinecone_client()
    # add date from now() formatted mm-dd-yyyy
    index_name = f"arxiv-index-{datetime.now().strftime('%m-%d-%Y')}"
    index = get_or_create_index(pc, index_name, arxiv_embedding.EMBEDDING_DIM)
    metadatas, embeddings = arxiv_embedding.embed_metadata("machine learning", 500, load_cache=False)
    """
    eg:
     {
      "id": "A", 
      "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
      "metadata": {"genre": "comedy", "year": 2020}
    }
    """
    # zip into vectors list of embeddings and metadata
    vectors = map(get_record, embeddings, metadatas)
    index.upsert(
        vectors=vectors,
        namespace="arxiv-metadata",
    )
    print(f"Index {index_name} has been updated with {len(metadatas)} papers.")

def build_semanticscholar_offline_dataset(limit=1000000, min_citation_count=30):
    try:
        ds = datasets.load_from_disk(f"semanticscholar_dataset_mincitation{min_citation_count}")
        return ds
    except FileNotFoundError:
        pass

    sch = SemanticScholarWrapper()
    def gen(shards):
        i = 0
        for year in shards:
            for x in sch.bulk_search(query="", fields_of_study=["Computer Science"], year=str(year), min_citation_count=min_citation_count):
                if i > limit:
                    break
                i += 1
                yield dict(x)
    ds = datasets.Dataset.from_generator(gen, gen_kwargs={
        "shards": list(range(2018, 2025)),
    }, num_proc=4)
    ds.save_to_disk(f"semanticscholar_dataset_mincitation{min_citation_count}")
    return ds


def get_semanticscholar_record(embedding_model_name):
    def map_record(row):
        filtered_metadata_columns = ["paperId", "title", "year", "citationCount", 'abstract']
        metadata = {k: v for k, v in row.items() if k in filtered_metadata_columns}
        return {
            "id": f"{row['paperId']}#semanticscholar-metadata#{embedding_model_name}",
            "values": row['embeddings'],
            "metadata": metadata
        }
    return map_record

def semanticscholar_embedding_content(row):
    content = row['title']
    if 'abstract' in row and row['abstract']:
        content += " " + row['abstract']
    return content

def embed_batch(embedding_model, b):
    ds = datasets.Dataset.from_dict(b)
    sch = SemanticScholarWrapper()    
    embeddings = sch.get_specter_embeddings(ds['paperId'])
    #embeddings = embedding_model.embed_parallel([semanticscholar_embedding_content(row) for row in ds])
    return { "embeddings": list(embeddings) }

def batch_generator(dataset, batch_size=500):
    """Lazy generator for creating batches from the dataset."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

def worker_init(index_name, dim):
    global index
    pc = get_pinecone_client()
    index = get_or_create_index(pc, index_name, dim, metric=METRIC_L2)
 
def index_fn(embedding_model, b):
    global index
    ds = datasets.Dataset.from_dict(b)
    vectors = map(get_semanticscholar_record(embedding_model), ds)
    # skip missing embeddings
    filtered_vectors = filter(lambda x: x['values'] is not None, vectors)
    index.upsert(
        vectors=filtered_vectors,
        namespace="semanticscholar-metadata")

def build_semanticscholar_index(embedding_model):
    try:
        dataset = datasets.load_from_disk("semanticscholar_dataset_mincitation30")
    except FileNotFoundError:
        dataset = build_semanticscholar_offline_dataset()
        #dataset.with_format("torch")
        #dataset = dataset.map(semanticscholar_embedding_content)
        #dataset = dataset.map(lambda b: {'embeddings': embed_parallel(b['content']) }, batched=True, batch_size=250)
        def embed_fn(b):
            return embed_batch(embedding_model, b)
        dataset = dataset.map(embed_fn, batched=True, batch_size=500, num_proc=4)
        dataset.save_to_disk("semanticscholar_embeddings_dataset")

 
    #dataset.map(index_fn, batched=True, batch_size=500)
    # rewrite using concurrent indexes (index not pickable, so cant use .map)


    workers = 4
    index_name = f"semanticscholar-index-{embedding_model.embedding_model()}-{datetime.now().strftime('%m-%d-%Y')}"
    dim = embedding_model.embedding_dim()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(index_name, dim)) as executor:
        batch_size = 500
        # Use tqdm for the progress bar
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        futures = []
        with tqdm(total=total_batches) as pbar:
            def wait():
                # Update progress bar and checkpoint after processing
                while futures:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for done_future in done:
                        pbar.update(1)
                        futures.remove(done_future)

            for batch in batch_generator(dataset, batch_size):
                # Submit the batch indexing to the executor
                future = executor.submit(index_fn, embedding_model.embedding_model(), batch)
                futures.append(future)

                if len(futures) < workers * 2:
                    continue

                wait()
            wait()

        

if __name__ == "__main__":
    #build_arxiv_index()
    embedding_model = Specter2Document()
    build_semanticscholar_index(embedding_model)
