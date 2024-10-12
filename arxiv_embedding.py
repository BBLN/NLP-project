import arxiv
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

EMBEDDING_DIM = 768
# Load a pre-trained NLP model (SciBERT for scientific text embedding)
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
client = arxiv.Client()

def fetch_arxiv_metadata(query, max_results=500):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    metadata = []
    for result in client.results(search):
        metadata.append({
            'id': result.entry_id,
            'categories': result.categories,
            'title': result.title,
            'abstract': result.summary,
        })
    return pd.DataFrame(metadata)


def embed_text(text):
    """
    Embed text (metadata) into vector space
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the CLS token embedding as a representation of the text
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings


def embed_metadata(query, max_results):
    """
    Embed arxiv metadata (title and abstract) into a vector space, and save them into pt file
    :param query: Subject of papers to embed
    :param max_results: Number of papers to embed
    """
    # Check if the embeddings have already been computed
    try:
        all_embeddings = torch.load(f"arxiv_metadata_embeddings_{query}.pt")
        metadatas = pd.read_csv(f"arxiv_metadata_{query}.csv").to_dict(orient='records')
        return metadatas, all_embeddings
    except FileNotFoundError:
        pass
    
    # Fetch metadata from the ArXiv API
    df = fetch_arxiv_metadata(query, max_results)

    # Embed the titles and abstracts into vector space
    embeddings = []
    metadatas = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Combine title and abstract to embed them together
        text_to_embed = row['title'] + " " + row['abstract']
        embedding = embed_text(text_to_embed)
        embeddings.append(embedding)
        metadatas.append(row.to_dict())

    # Convert embeddings list to a tensor
    all_embeddings = torch.cat(embeddings)

    # Save the embeddings to disk for future use
    torch.save(all_embeddings, f"arxiv_metadata_embeddings_{query}.pt")

    # Save metadatas
    pd.DataFrame(metadatas).to_csv(f"arxiv_metadata_{query}.csv", index=False)

    return metadatas, all_embeddings


if __name__ == '__main__':
    embed_metadata("hypergraphs", 10)