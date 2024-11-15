# --- START OF FILE trainer.py ---
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM, TextGenerationPipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import AutoModelForCausalLMWithValueHead, PPOv2Trainer, PPOv2Config, PPOTrainer, PPOConfig
from typing import List, Dict, Tuple
from tqdm import tqdm
import random
import semanticscholar
from semanticscholar.Paper import Paper
import wandb
from api_keys import WANDB_API_KEY
import datasets
from datasets import Dataset
import argparse
from datetime import datetime
import traceback
import concurrent
from transformers.pipelines.pt_utils import KeyDataset

wandb.login(key=WANDB_API_KEY)
wandb.init()

sch = semanticscholar.SemanticScholar()

def get_random_papers_from_semantic_scholar(num_papers: int):
    """Fetches random papers from Semantic Scholar."""
    #papers = list(sch.search_paper(query="network", fields_of_study=["Computer Science"], limit=100, min_citation_count=20, bulk=False))
    papers = list(sch.search_paper(query="", fields_of_study=["Computer Science"], year='2016-', bulk=True, min_citation_count=30,
                  fields=['title', 'paperId', 'citationCount', 'authors', 'abstract', 'publicationDate', 'year', 'influentialCitationCount']))
    # shuffle paperIds
    #random.shuffle(papers)
    pbar = tqdm(total=len(papers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #for paper, citations in tqdm(executor.map(lambda paper: (dict(paper), list(sch.get_paper_citations(paper.paperId, fields=["contexts", "citingPaper"]))), papers), total=len(papers)):
        #    yield {"paper": paper, "citingPaper": citations.paper(), "excerpts": citations['contexts']}
        def fetch_fn(paper):
            try:
                res = list(sch.get_paper_citations(paper.paperId, fields=["contexts", "citingPaper.paperId", "citingPaper.title"]))
                pbar.update(1)
                return list(map(lambda citations: {"paper": dict(paper), "citingPaper": dict(citations.paper), "excerpts": citations['contexts']}, res))
            except Exception as e:
                print(f"Error fetching paper {paper.paperId}: {e}")
                traceback.print_exc()
                return []
        futures = []
        for paper in papers:
            futures.append(executor.submit(fetch_fn, paper))
        for future in concurrent.futures.as_completed(futures):
            yield from future.result()
            


def search_papers(query: str, top_k: int = 10) -> Tuple[List[str], int]:
    """Searches for papers and returns titles and total matches."""
    search = sch.search_paper(query=query, limit=top_k, fields=['title','paperId'], sort='citationCount:desc')
    results = list(search)
    return results, search.total

def create_training_data(num_papers: int):
    for res in get_random_papers_from_semantic_scholar(num_papers):
        yield res

def init_dataset():
    try:
        ds = datasets.load_from_disk("ppo_trainset2")
    except:
        training_data = create_training_data(DATASET_SIZE)
        ds = Dataset.from_generator(training_data)
        ds.save_to_disk("ppo_trainset2")
    return ds

# --------------------------------------------------------

# --- Configuration ---
MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
#MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

LOAD_IN_4BIT = True
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune a model for query generation.")
parser.add_argument("--model_name", type=str, default="microsoft/phi-3.5-mini-instruct", help="Name of the model to use.")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--padding_side", type=str, default="left", help="Padding side for the tokenizer.")
args = parser.parse_args()

MODEL_NAME = args.model_name
DATASET_SIZE = 32 * 5
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

import uuid

class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")

def create_dataset():
    try:
        ds = Dataset.from_generator(_DatasetGeneratorPickleHack(lambda: create_training_data(32000), generator_id=f"citations-{current_time}"))
        ds.save_to_disk("citations")
    except:
        print("Error creating dataset")
        traceback.print_exc()
    return ds

try:
    ds = datasets.load_from_disk("citations")
except:
    create_dataset()

SHARDS = 8
try:
    ds = datasets.load_from_disk(f"citations2_ds_phi_ranked_shard{SHARDS-1}")
except:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # --- Model and Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer.model_max_length = 2048
    """
    if 'phi' in MODEL_NAME:
        tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
    else:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = args.padding_side
    """

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                quantization_config=quantization_config,
                                                device_map="auto")
    print(model.device)
    pipeline = TextGenerationPipeline(model, tokenizer, max_new_tokens=50, max_length=None)

    # Evaluate dataset
    def build_eval_prompt(batch):
        results = []
        for i in range(len(batch['paper'])):
            paper = batch['paper'][i]
            excerpts = batch['excerpts'][i]
            for excerpt in excerpts:
                messages = [
                    { 'role' : 'user', 'content': "For the given cited paper and excerpts, respond with YES or NO if the excerpt is attributable to the cited paper title. Then explain in brief why, come up with a query from the excerpt that would've returned that paper\n" \
                    f"Cited paper title: {paper['title']}\n" \
                    f"Excerpts: {excerpt[:256]}\n" },
                    { 'role' : 'assistant', 'content': "ANSWER: "}
                ]
                results.append({ 'title': paper['title'], 'paperId': paper['paperId'], 'prompt': messages, 'excerpt': excerpt })
        return {
            'title': [x['title'] for x in results],
            'paperId': [x['paperId'] for x in results],
            'prompt': [x['prompt'] for x in results],
            'excerpt': [x['excerpt'] for x in results]
        }

    # check which excerpts the model thinks are unattributable
    def eval_excerpt(b):
        output = pipeline(b, continue_final_message=True, return_full_text=False)
        print(output)
        return output

    #ds.filter(lambda example, idx: idx < 100 and len(example['excerpts']) > 0, with_indices=True).map(eval_excerpt)

    ds = ds.map(build_eval_prompt, batched=True, remove_columns=ds.column_names)
    # add output row using batched pipeline
    #for out in pipeline(KeyDataset(ds, 'prompt'), continue_final_message=True, return_full_text=False, batch_size=32):
    #ds.add_column(out, 'output')
    # sharding to split work to checkpoints

    def eval_prompt(b):
        res = {
            'eval': [x[0]['generated_text'] for x in pipeline(b['prompt'], continue_final_message=True, return_full_text=False, batch_size=32)]
        }
        print(b['prompt'][0], res['eval'][0])
        return res
    for shard_idx in range(SHARDS):
        shardds = ds.shard(num_shards=SHARDS, index=shard_idx)
        shardds = shardds.map(eval_prompt, batched=True, batch_size=1024)
        shardds.save_to_disk(f"citations2_ds_phi_ranked_shard{shard_idx}")

# join shards back with from_generator

def join_shards():
    shards = [datasets.load_from_disk(f"citations2_ds_phi_ranked_shard{i}") for i in range(SHARDS)]
    return datasets.concatenate_datasets(shards)

try:
    ds = datasets.load_from_disk("citations2_ds_phi_ranked")    
except:
    ds = join_shards()
    ds.save_to_disk("citations2_ds_phi_ranked")

ds = ds.filter(lambda b: [x.startswith(' YES') for x in b['eval']], batched=True, batch_size=256)
ds.save_to_disk("citations2_ds_phi_ranked_attributable")