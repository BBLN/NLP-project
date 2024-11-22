import torch
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, Adafactor, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from api_keys import WANDB_API_KEY
import wandb
from datasets import load_from_disk
import argparse
from datetime import datetime
import random
from tqdm import tqdm
import json
from peft import PeftModel

wandb.login(key=WANDB_API_KEY)
wandb.init(project="finetune-select-ppo")

LOAD_IN_4BIT = True
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune a model for query generation.")
parser.add_argument("--model_name", type=str, default="microsoft/phi-3.5-mini-instruct", help="Name of the model to use.")
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
LORA_OUTPUT_DIR = f"./ppo-select-{MODEL_NAME.replace('/', '-')}_{current_time}"

def load_model_unsloth():
    from unsloth import FastLanguageModel
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "sft-3-epochs-86", # pretrained for queries, original "unsloth/Phi-3.5-mini-instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
    # target_modules = 'all-linear',
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return AutoModelForCausalLMWithValueHead(model), tokenizer

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.model_max_length = 2048
    if 'phi' in MODEL_NAME:
        tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
    else:
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

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
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        ))
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, 'sft-3-epochs-86', is_trainable=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    return model, tokenizer

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

def init_trainer(model, tokenizer):
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

def format_instructions():
    return 'Select cited paper: {"reason": "why", "action": {"name": "select_paper", "paper_id": "paper ID"}}\n'

import csv
gold_f = open(f'gold_selections_{current_time}.csv', 'w')
gold_csv = csv.DictWriter(gold_f, fieldnames=['paperId', 'title', 'excerpt', 'reason'])
gold_csv.writeheader()


def train_batch(ppo_trainer, tokenizer, selections):
    # --- Training Loop ---
    generation_kwargs = {
        "min_length": -1,
        "max_new_tokens": 128,
        #"num_beams": 2,
        "do_sample": True,
        "top_k": 2,
        #"top_p": 0.7,
        'pad_token_id': tokenizer.eos_token_id,
    }


    batch = dict(query=selections, response=[])
    inputs = []

    inputs = []
    cited_papers = []
    excerpts = []
    for item in batch['query']:
        excerpt = item['excerpt']
        papers = item['results'][:10]
        papers_str = ""
        for paper in papers:
            papers_str += f"- Paper ID: {paper['paperId']}\n"
            papers_str += f"\tTitle: {paper['title']}\n"
            if paper['abstract']:
                papers_str += f"\tAbstract: {paper['abstract'][:128]}\n"
            papers_str += f"\tCitation Count: {paper['citationCount']}\n\n"

        messages = [
            {"role": "user", "content": f"I'm looking for a paper referenced in this excerpt:\n\n{excerpt}\n\nOut of the following paper list, you must select single paper ID as instructed\n\n{papers_str}\n\nUse JSON format as follows:\n{format_instructions()}" },
        ]
        excerpts.append(excerpt)
        inputs.append(messages)
        cited_papers.append(dict(paperId=item['paperId'], title=item['title']))

    # Get response from model
    query_tensors = tokenizer.apply_chat_template(inputs, return_tensors="pt", padding=True, add_generation_prompt=True).to("cuda")
    start_time = datetime.now()
    ppo_trainer.model.gradient_checkpointing_disable()
    response_tensors, response_ref_tensors = ppo_trainer.generate(list(query_tensors), return_prompt=False, generate_ref_response=True, batch_size=args.batch_size, **generation_kwargs)
    ppo_trainer.model.gradient_checkpointing_enable()
    wandb.log({"generation_time": (datetime.now() - start_time).total_seconds()})
    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    ref_responses = tokenizer.batch_decode(response_ref_tensors, skip_special_tokens=True)

    def try_reward(action, ref_action, p, e):
        print(f"\naction: {action}\nref action: {ref_action}\npaper: {p['title']}\nexcerpt: {e}")
        try:
            action = json.loads(action)
            selectedPaper = action['action']['paper_id']
            action = action['action']['name']
            reason = action['reason']
        except:
            print("Invalid action")
            return -5
        if action != "select_paper":
            print("Invalid action")
            return -5
        if selectedPaper != p['paperId']:
            print("Incorrect paper selected")
            return -1
        gold_csv.writerow({'paperId': p['paperId'], 'title': p['title'], 'excerpt': e, 'reason': reason})
        return 1
         
    # log wandb time duration to compute rewards
    start_time = datetime.now()
    rewards = []
    for q, q_ref, paper, e in zip(responses, ref_responses, cited_papers, excerpts):
        rewards.append(torch.tensor([float(try_reward(q, q_ref, paper, e))]))
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

def train_ppo():
    dataset = load_from_disk("queries_dataset_ranked_20241115_231741")
    dataset = dataset.filter(lambda x: x["relevance_rank"] <= 10 or x["citation_rank"] <= 10)

    model, tokenizer = load_model()
    trainer = init_trainer(model, tokenizer)

    training_data = list(dataset)
    for epoch in tqdm(range(args.epochs), "epoch: "):         
        # shuffle training data
        random.shuffle(training_data)

        # chunk training data to batch size
        batches = [training_data[i:i + ppo_config.batch_size] for i in range(0, len(training_data), ppo_config.batch_size)]
        # drop last batch if it is smaller than batch size
        if len(batches[-1]) < ppo_config.batch_size:
            batches = batches[:-1]

        for queriesBatch in tqdm(batches, "Training: "):
            train_batch(trainer, tokenizer, queriesBatch)
        trainer.save_pretrained(f"{LORA_OUTPUT_DIR}-epoch{epoch}")

if __name__ == "__main__":
    train_ppo()