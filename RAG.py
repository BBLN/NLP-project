#from arxiv_embedding import embed_text, EMBEDDING_DIM
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from pinecone_db import get_pinecone_client, get_or_create_index
from specter_embedding import Specter2AdhocQuery

class RAGRetriever:
    def __init__(self):
        self.pc = get_pinecone_client()
        self.index_name = "semanticscholar-index-specter2-10-28-2024"
        self.embedding_model = Specter2AdhocQuery()
        self.index = get_or_create_index(self.pc, self.index_name, self.embedding_model.embedding_dim())

    def retrieve_relevant_papers(self, query, top_k=3):
        query_embedding = self.embedding_model.embed_parallel(query).squeeze()
        vectors = self.index.query(vector=query_embedding.tolist(), namespace="semanticscholar-metadata", top_k=top_k, include_metadata=True)
        return vectors
    

def get_paper_citation_by_title(title, number_of_citations=10):
    import arxiv
    from pylatexenc.latex2text import LatexNodes2Text
    # Search for the paper using its title
    res = []
    search = arxiv.Search(
        query=title,
        max_results=number_of_citations,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = list(search.results())

    if not results:
        return None, "No paper found with the given title."

    for paper in results:
        # Extract relevant information
        title = LatexNodes2Text().latex_to_text(paper.title)
        authors = ", ".join([author.name for author in paper.authors])
        year = paper.published.year
        arxiv_id = paper.get_short_id()
        url = paper.entry_id

        # Generate the citation
        citation = f"{authors}. ({year}). {title}. arXiv:{arxiv_id}. Retrieved from {url}"
        res.append((title, citation))
    return res


def generate_answer(query, titles, pipeline):
    """
    Generate an answer based on retrieved Arxiv papers.
    """

    context = ""
    # Append the title and citations of the papers to form a context
    for title in titles:
        citations = get_paper_citation_by_title(title)
        context += f"Title: {title}\n"
        for cite in citations:
            context += f"citations: {cite[0]}, {cite[1]}\n\n"

    # Prompt model to generate a response using the retrieved context
    input_prompt = f"What is the paper {query} citations, based on the next papers and citations: {context}\n"

    # # Generate a response
    sequences = pipeline(
        input_prompt,
        do_sample=True,
        top_k=50,
        top_p=0.7,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=500,
    )

    return sequences


def find_citetion_of_query(query):
    model = "PY007/TinyLlama-1.1B-Chat-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # create the indexes or use one allready exist
    index = None
    titles = retrieve_relevant_papers(query, index)
    sequences = generate_answer(query, titles, pipeline)
    filename = f"citetions of {query}"
    with open(filename, 'w') as file:
        file.write(sequences['generated_text'] + '\n')

PROMPT_TEMPLATE = "**You are given an excerpt from a paper, where a citation was deleted. I'm trying to find the citation (ignore the word [CITATION], that's just where the citation was deleted from. You will be asked to help me find the paper from which the citation was deleted.\n" \
    "You can ask me a search query and I will try to find related papers. I can't give you more data from the paper, or any data about the cited paper. You have to do it exactly once. After that, you will be asked to provide the citation. If you don't know the citation, you can say 'I don't know'.**\n" \
    "EXCERPT START\n{excerpt}\nEXCERPT END\n\n**Now just supply the query in this format: Query: text**"

class CitationFinder:
    def __init__(self):
        model_name = "microsoft/Phi-3.5-mini-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.retriever = RAGRetriever()

    def ask_model(self, *input):
        messages = [
            { "role": "system", "content": "You are a helpful AI assistant." },
        ]
        for i, message in enumerate(input):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({ "role": role, "content": message })
        
        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        return self.pipeline(messages, **generation_args)[0]['generated_text']
        
    def find_citation(self, excerpt):
        messages = [PROMPT_TEMPLATE.format(excerpt=excerpt)]
        search_query = self.ask_model(*messages)
        messages.append(search_query)
        search_query = search_query.removeprefix("Query: ")
        print(search_query)
        related_papers = self.retriever.retrieve_relevant_papers(search_query)
        print(related_papers)
        titles = [paper['metadata']['title'] for paper in related_papers['matches']]
        select_paper_message = '\n'.join([f"{i+1}. {title}" for (i, title) in enumerate(titles)])
        if not related_papers['matches']:
            select_paper_message = 'No papers found'
        messages.append(select_paper_message + '\nNow just reply with your guessed cited paper title')
        print(select_paper_message)
        citation = self.ask_model(*messages)
        print(citation)
        return citation, messages
