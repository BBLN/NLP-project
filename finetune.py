# --- START OF FILE trainer.py ---
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import AutoModelForCausalLMWithValueHead, PPOv2Trainer, PPOv2Config, PPOTrainer, PPOConfig
from typing import List, Dict, Tuple
from tqdm import tqdm
import random 
import semanticscholar_wrapper
import semanticscholar
import wandb

wandb.init()

sch = semanticscholar_wrapper.SemanticScholarWrapper()
# --- Replace these with your actual implementations ---
def get_random_papers_from_semantic_scholar(num_papers: int) -> List[Tuple[semanticscholar.Paper, str]]:
    """Fetches random papers from Semantic Scholar."""
    papers = list(sch.bulk_search(query="", fields_of_study=["Computer Science"], limit=num_papers, min_citation_count=10))
    # shuffle paperIds
    random.shuffle(papers)

    results = []
    for paper in papers:
        citations = sch.client.get_paper_citations(paper.paperId, fields=["contexts"])
        for citation in citations:
            for excerpt in citation.contexts:
                if len(results) >= num_papers:
                    return results
                results.append((paper, excerpt))
    return results


def search_papers(query: str, top_k: int = 10) -> Tuple[List[str], int]:
    """Searches for papers and returns titles and total matches."""
    search = sch.search(query=query, limit=top_k)
    results = list(search)
    return results, search.total

print(search_papers('limitations low signal-to-noise ratios minimal radiation contrast interference surrounding structures catheters bones reference 12 34'))
# --------------------------------------------------------

# --- Configuration --- 
MODEL_NAME = "microsoft/phi-3.5-mini-instruct"  
LOAD_IN_4BIT = True  
DATASET_SIZE = 100  
LORA_OUTPUT_DIR = "./lora-query-generator" 

# --- Data Preparation ---
def create_training_data(num_papers: int):
    papers = get_random_papers_from_semantic_scholar(num_papers)
    data = []
    for paper, excerpt in tqdm(papers, desc="Processing papers"):
        data.append({"excerpt": excerpt, "cited": paper})
    return data

# --- Reward Function ---
def compute_reward(query: str, target_citation: Dict, top_k: int = 10) -> float:
    print(query, target_citation.title)
    results, total_matches = search_papers(query, top_k=top_k) 
    target_title = target_citation['title']

    titles = list(map(lambda p: p['title'], results))
    if target_title in titles:
        rank = titles.index(target_title) + 1 
        rank_reward = (top_k - rank + 1) / top_k  
        specificity_threshold = 0.05
        specificity_reward = 1.0 if (rank <= top_k * specificity_threshold) else max(0, 1 - (rank / (total_matches * specificity_threshold)))
        combined_reward = 0.8 * rank_reward + 0.2 * specificity_reward
        return combined_reward
    else:
        return 0.0 

# --- Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=1,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

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
    )
    #print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))
    #model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':True})
    #print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, peft_config=lora_config)

#model = get_peft_model(model, lora_config)
#model.print_trainable_parameters()

# --- PPO Configuration ---
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=3e-4,
    batch_size=16,
    mini_batch_size=4,
    log_with=['wandb']
)

# --- PPOTrainer Initialization ---
ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer, dataset=None) 

# --- Training Loop ---
training_data = create_training_data(DATASET_SIZE)
epochs = 10
generation_kwargs = {
    "min_length": 10,
    "max_new_tokens": 200,
    "num_beams": 4,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    'pad_token_id': tokenizer.eos_token_id,
}

# --- Prompt Template (as a list of messages) ---
SYSTEM_PROMPT = "You are a helpful AI assistant that generates search queries to find scientific papers."
PROMPT_TEMPLATE = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Generate a search query to find the paper being cited in the following excerpt:\n\nExcerpt: {excerpt}\n\nRespond with a JSON such as \{\"query\":\"string\"\}"}
]

for epoch in tqdm(range(epochs), "epoch: "):
    # shuffle training data
    random.shuffle(training_data)
    
    # chunk training data to batch size
    batches = [training_data[i:i + ppo_config.batch_size] for i in range(0, len(training_data), ppo_config.batch_size)]
    # drop last batch if it is smaller than batch size
    if len(batches[-1]) < ppo_config.batch_size:
        batches = batches[:-1]

    for queriesBatch in tqdm(batches, "Training: "):
        batch = dict(query=queriesBatch, response=[])
        queries = []
        responses = []
        rewards = []

        inputs = []
        outputs = []
        for item in batch['query']:
            excerpt = item['excerpt']
            target_paper = item['cited']
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a search query to find the paper being cited in the following excerpt:\n\nExcerpt: {excerpt}\n\nRespond with query without any quotation or punctuation" },
                {"role": "assistant", "content": 'Query: '}
            ]
            inputs.append(messages)
            outputs.append(target_paper)

        # Get response from model
        query_tensors = tokenizer.apply_chat_template(inputs, continue_final_message=True, return_tensors="pt", padding=True).to("cuda")
        response_tensors = ppo_trainer.generate(list(query_tensors), return_prompt=False, **generation_kwargs)
        responses = tokenizer.decode(response_tensors, skip_special_tokens=True)
        rewards = map(lambda r, p: compute_reward(r, p), responses, outputs)
        """
        for item in batch['query']:
            excerpt = item['excerpt']
            target_paper = item['cited']

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a search query to find the paper being cited in the following excerpt:\n\nExcerpt: {excerpt}\n\nRespond with query without any quotation or punctuation" },
                {"role": "assistant", "content": 'Query: '}
            ]
            query_tensors = tokenizer.apply_chat_template(messages, continue_final_message=True, return_tensors="pt", padding=True).to("cuda")
            #query_tensors = tokenizer(inputs, return_tensors="pt").input_ids.to("cuda")

            # Get response from model
            response_tensors = ppo_trainer.generate(query_tensors.squeeze(), return_prompt=False, **generation_kwargs)
            response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
            print('\n\n' + response_text)
            batch['response'].append(response_text)
            response = response_text
            #try:
            reward = compute_reward(response, target_paper)
            print(reward)
            #except Exception as e:
            #    print(e)
            #    reward = -1.0
            wandb.log({"item_reward": reward})
            queries.append(query_tensors[0])
            responses.append(response_tensors[0])
            rewards.append(torch.tensor([reward]))
        """

        # Run PPO step
        stats = ppo_trainer.step(queries, responses, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# --- Save the LoRA model ---
ppo_trainer.save_pretrained(LORA_OUTPUT_DIR)