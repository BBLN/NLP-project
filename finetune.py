# --- START OF FILE trainer.py ---
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Adafactor
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
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
import math
import urllib.parse
import json
import random

wandb.login(key=WANDB_API_KEY)
wandb.init(project="finetune-query-generator")

sch = semanticscholar.SemanticScholar()

def get_random_papers_from_semantic_scholar(num_papers: int) -> List[Tuple[Paper, str]]:
    """Fetches random papers from Semantic Scholar."""
    #papers = list(sch.search_paper(query="network", fields_of_study=["Computer Science"], limit=100, min_citation_count=20, bulk=False))
    papers = list(sch.search_paper(query="", fields_of_study=["Computer Science"], year='2016-', bulk=True, min_citation_count=20))
    # shuffle paperIds
    random.shuffle(papers)

    results = []
    with tqdm(papers, desc="Collecting citations", total=num_papers) as pbar:
      for paper in papers:
          citations = sch.get_paper_citations(paper.paperId, fields=["contexts"])
          # limit number of citations of same paper
          i = 0
          for citation in citations:
              if i >= 5:
                  break
              i += 1
              for excerpt in citation.contexts:
                  if len(results) >= num_papers:
                      return results
                  pbar.update(1)
                  results.append((dict(paper), excerpt))
    return results


def search_papers(query: str, sortByCitations=False) -> Tuple[List[str], int]:
    """Searches for papers and returns titles and total matches."""
    sortBy = None
    if sortByCitations:
        sortBy = 'citationCount:desc'
    search = sch.search_paper(query=query, fields=['paperId', 'citationCount'], limit=100)
    results = list(search)
    if sortByCitations:
        results = sorted(results, key=lambda x: x['citationCount'], reverse=True)
    return results, search.total

def create_training_data(num_papers: int):
    papers = get_random_papers_from_semantic_scholar(num_papers)
    data = []
    for paper, excerpt in tqdm(papers, desc="Processing papers"):
        data.append({"excerpt": excerpt, "cited": paper})
    return data

def init_dataset():
    try:
        ds = datasets.load_from_disk("citations2_ds_phi_ranked_attributable")
        # filter out long excerpts to prevent OOM
        ds = ds.filter(lambda x: len(x['excerpt'] + x['title']) < 350)
        ds = ds.map(lambda x: {"cited": dict(paperId=x['paperId'], title=x['title']), 'excerpt':x['excerpt']}, remove_columns=ds.column_names)
    except:
        try:
            ds = datasets.load_from_disk("ppo_trainset2")
        except:
            training_data = create_training_data(DATASET_SIZE)
            ds = Dataset.from_list(training_data)
            ds.save_to_disk("ppo_trainset2")
    return ds

# --------------------------------------------------------

# --- Configuration ---
#MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct" # "meta-llama/Meta-Llama-3-8B-Instruct"

LOAD_IN_4BIT = True
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune a model for query generation.")
parser.add_argument("--model_name", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct", help="Name of the model to use.")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
parser.add_argument("--ppo_batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--ppo_minibatch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--padding_side", type=str, default="left", help="Padding side for the tokenizer.")
parser.add_argument("--adafactor", action="store_true", default=False, help="Use Adafactor optimizer.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
args = parser.parse_args()

MODEL_NAME = args.model_name
DATASET_SIZE = 32 * 5
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LORA_OUTPUT_DIR = f"./lora-query-generator-{MODEL_NAME.replace('/', '-')}_{current_time}"

# --- Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 512
if 'phi' in MODEL_NAME:
    tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
else:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
#tokenizer.padding_side = args.padding_side

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules='all-linear',
    modules_to_save=None,
    #target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

def load_model():
    if LOAD_IN_4BIT:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            peft_config=lora_config,
            #attn_implementation="flash_attention_2",
        )
        #print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))
        #model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':True})
        #print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, peft_config=lora_config) #attn_implementation="flash_attention_2",)
    #model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters()
    return model

# --- PPO Configuration ---
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1e-5,
    batch_size=args.ppo_batch_size,
    mini_batch_size=args.ppo_minibatch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    log_with=['wandb']
)

def init_trainer(model):
    # --- PPOTrainer Initialization ---
    optimizer = None
    if args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=ppo_config.learning_rate,
        )
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer, dataset=None, optimizer=optimizer)
    return ppo_trainer

# --- Training Loop ---
generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": 256,
    #"num_beams": 2,
    "do_sample": True,
    #"top_k": 0,
    "top_p": 0.6,
    'pad_token_id': tokenizer.eos_token_id,
}

# --- Prompt Template (as a list of messages) ---
SYSTEM_PROMPT = "You are a helpful AI assistant that generates search queries to find scientific papers.\n" \
"for example, generate the query 'modality gap geometry' for excerpt: Specifically, although [5] first demonstrated that there exists a modality gap between text and image embeddings generated from VLMs, the geometry of this modality gap permits cross-modality transferability. This phenomenon allows text to serve as a proxy to corresponding images and vice versa."
#PROMPT_TEMPLATE = [
#    {"role": "system", "content": SYSTEM_PROMPT},
#    {"role": "user", "content": "Generate a search query to find the paper being cited in the following excerpt:\n\nExcerpt: {excerpt}\n\nRespond with a JSON such as \{\"query\":\"string\"\}"}
#]

# --- Reward Function ---

import csv
gold_queries_f = open(f'gold_queries_{current_time}.csv', 'w')
gold_queries = csv.DictWriter(gold_queries_f, fieldnames=['paperId', 'title', 'query', 'excerpt','reason'])
gold_queries.writeheader()

def compute_reward(query: str, target_citation: Dict, excerpt, reason, sort, top_k: int = 1000) -> float:
    #print(query, target_citation['title'])
    results, total_matches = search_papers(urllib.parse.quote(query), sortByCitations=(sort == "search_citation_count"))
    target_id = target_citation['paperId']

    if total_matches == 0:
        return 0.1
    specificity_reward = 1 / math.log(total_matches + 1) + 0.3
    ids = list(map(lambda p: p['paperId'], results))
    if target_id in ids:
        gold_queries.writerow({'paperId': target_citation['paperId'], 'query': query, 'title': target_citation['title'], 'excerpt': excerpt, 'reason': reason})
        gold_queries_f.flush()

        rank = ids.index(target_id) + 1
        print("paper found!", rank)

        specificity_reward = 1 / total_matches
    
        combined_reward = 10 / math.sqrt(rank) + specificity_reward
        return combined_reward
    return specificity_reward


ds = init_dataset()
model = load_model()
ppo_trainer = init_trainer(model)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=40)

def format_instructions():
    # rand bool
    show_citation_count = random.choice([True, False])
    show_relevance = random.choice([True, False])

    if not show_citation_count and not show_relevance:
        show_citation_count = True
    
    desc = f"You can use any of the following actions up to 5 times:\n"
    actions = []
    if show_citation_count:
        actions.append('Search by relevance: {"action": {"name": "search_relevance": "query": "search terms"}, "reason": "why"}\n')
    if show_relevance:
        actions.append('Search by citation count: {"action": {"name": "search_citation_count": "query": "search terms"}, "reason": "why"}\n')
    
    return desc + ''.join(actions)

def train_batch(queries):
    batch = dict(query=queries, response=[])
    responses = []
    rewards = []

    inputs = []
    outputs = []
    excerpts = []
    for item in batch['query']:
        excerpt = item['excerpt']
        target_paper = item['cited']
        messages = [
            #{"role": "system", "content": SYSTEM_PROMPT},
            #{"role": "user", "content": f"Generate a search query to find the paper being cited in the following excerpt:\n\nExcerpt: {excerpt}\n\nRespond with query without any quotation or punctuation" },
            {"role": "user", "content": f"I'm looking for a paper referenced in this excerpt:\n\n{excerpt}\n\nI need a short and concise query for the papers search engine. Avoid quotation marks, hyphens or any punctuation. Don't use extra spaces or period. Searching by author name or publication date is not allowed.\nUse JSON format as follows:\n{format_instructions()}" },
            #{"role": "assistant", "content": '{'}
        ]
        excerpts.append(excerpt)
        inputs.append(messages)
        outputs.append(target_paper)

    # Get response from model
    query_tensors = tokenizer.apply_chat_template(inputs, return_tensors="pt", padding=True, add_generation_prompt=True).to("cuda")
    start_time = datetime.now()
    ppo_trainer.model.gradient_checkpointing_disable()
    response_tensors, response_ref_tensors = ppo_trainer.generate(list(query_tensors), return_prompt=False, generate_ref_response=True, batch_size=args.batch_size, **generation_kwargs)
    ppo_trainer.model.gradient_checkpointing_enable()
    wandb.log({"generation_time": (datetime.now() - start_time).total_seconds()})
    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    ref_responses = tokenizer.batch_decode(response_ref_tensors, skip_special_tokens=True)

    def try_reward(action, ref_q, p, e):
        action = action
        print(f"\n\nquery: {action}\nref query: {ref_q}\npaper: {p['title']}\nexcerpt; {e}")
        try:
            action = json.loads(action)
            q = action['action']['query']
            sort = action['action']['name']
            reason = action['reason']
        except:
            print("invalid json")
            return -1.0
        q = q.strip()
        if len(q) == 0 or '\n' in q:
            return -0.3 # penalize illegal empty or multiline queries
        # count english letters >= 2
        if sum(map(lambda c: c.isalpha(), q)) < 2:
            return -0.3
        try:
            reward = compute_reward(q, p, e, reason, sort)
            print(reward)
            return reward
        except Exception as e:
            print(traceback.format_exc())
            return -0.3
    # log wandb time duration to compute rewards
    start_time = datetime.now()
    futures = []
    for q, q_ref, p, e in zip(responses, ref_responses, outputs, excerpts):
        futures.append(executor.submit(try_reward, q, q_ref, p, e))
    rewards = [torch.tensor([f.result()]) for f in futures]
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    wandb.log({"reward_computation_duration": duration})

    total_rewards = sum(rewards)
    print(total_rewards)
    wandb.log({"reward": total_rewards})
    # Run PPO step
    start_time = datetime.now()
    stats = ppo_trainer.step(list(query_tensors), list(response_tensors), rewards)
    wandb.log({"step_time": (datetime.now() - start_time).total_seconds()})
    ppo_trainer.log_stats(stats, batch, rewards)
    
training_data = list(ds)
training_data = training_data[:10000]

for epoch in tqdm(range(args.epochs), "epoch: "):
    # shuffle training data
    random.shuffle(training_data)

    # chunk training data to batch size
    batches = [training_data[i:i + ppo_config.batch_size] for i in range(0, len(training_data), ppo_config.batch_size)]
    # drop last batch if it is smaller than batch size
    if len(batches[-1]) < ppo_config.batch_size:
        batches = batches[:-1]

    for queriesBatch in tqdm(batches, "Training: "):
        train_batch(queriesBatch)

# --- Save the LoRA model ---
ppo_trainer.save_pretrained(LORA_OUTPUT_DIR) #+ '_' + MODEL_NAME.replace('/', '-'))
