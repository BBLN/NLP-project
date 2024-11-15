import glob
import csv
import datasets
import semanticscholar
from datetime import datetime
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

# iglob all files with pattern gold_queries*.csv
def queries_files():
    return glob.iglob('gold_queries*.csv')

def csv_queries_iterator(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        # validate header is paperId,title,query,excerpt
        headers = reader.fieldnames
        if not headers:
            print(f"Empty file {fname}")
            return
        
        for h in ['paperId', 'title', 'query', 'excerpt', 'reason']:
            if h not in headers:
                print(f"Header {h} not found in {fname}")
                return
        for row in reader:
            yield row


def queries_dataset_join2():
    for fname in queries_files():
        for row in csv_queries_iterator(fname):
            yield row

def build_queries_dataset():
    ds = datasets.Dataset.from_generator(queries_dataset_join2)
    return ds

class SemanticScholarRanker:
    def __init__(self):
        self.sch = semanticscholar.SemanticScholar()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(20))
    def search_papers(self, query, sortByCitations=False):
        """Searches for papers and returns titles and total matches."""
        search = self.sch.search_paper(query=query, fields=['paperId', 'citationCount', 'title', 'abstract'], limit=30)
        results = list(search)
        if sortByCitations:
            results = sorted(results, key=lambda x: x['citationCount'], reverse=True)
        return results

    def rank_in_results(self, results, paperId):
        for i, res in enumerate(results):
            if res['paperId'] == paperId:
                return i
        return -1
    
    def paper_record(self, p):
        return {'paperId': p['paperId'], 'citationCount': p['citationCount'], 'title': p['title'], 'abstract': p['abstract'] }

    def rank_query(self, record):
        try:
            results = list(self.search_papers(record['query']))
            papers = [self.paper_record(paper) for paper in results]
            relevance_rank = self.rank_in_results(results, record['paperId'])
            results = sorted(results, key=lambda x: x['citationCount'], reverse=True)
            citation_rank = self.rank_in_results(results, record['paperId'])
            citations = -1 if citation_rank == -1 else results[citation_rank]['citationCount']
        except:
             return { 'relevance_rank': 100, 'citation_rank': 100, 'citations': -1, 'results': [] }
        #papers_str
        #for paper in results:
        #    papers_str += f"- Paper ID: {paper.paperId}\n"
        #    papers_str += f"\tTitle: {paper.title}\n"
        #    if paper.abstract:
        #        papers_str += f"\tAbstract: {paper.abstract[:256]}\n"
        #    papers_str += f"\tCitation Count: {paper.citationCount}\n\n"
        return {
            'relevance_rank': relevance_rank,
            'citation_rank': citation_rank,
            'citations': citations,
            'results': papers,
        }

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds = build_queries_dataset()
    ds.save_to_disk(f'queries_dataset_{current_time}')
    
    ds = ds.map(lambda x: SemanticScholarRanker().rank_query(x), num_proc=20)
    ds.save_to_disk(f'queries_dataset_ranked_{current_time}')
