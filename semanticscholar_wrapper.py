import semanticscholar

class SemanticScholarWrapper:
    def __init__(self):
        self.client = semanticscholar.SemanticScholar()

    def bulk_search(self, **kwargs):
        return self.client.search_paper(**kwargs, bulk=True)
    
    def get_specter_embeddings(self, paper_ids):
        papers = self.client.get_papers(paper_ids, fields=["embedding.specter_v2"])
        def get_embedding_vector(p):
            if not p.embedding:
                return None
            return p.embedding["vector"]
        # build dict paperId,embedding lazy
        paper_embeddings = {p.paperId: get_embedding_vector(p) for p in papers}
        return map(lambda x: paper_embeddings[x] if x in paper_embeddings else None, paper_ids)
