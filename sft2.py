import torch
from unsloth import FastLanguageModel
import wandb
from api_keys import WANDB_API_KEY
from datasets import load_from_disk, Dataset
from unsloth.chat_templates import get_chat_template
import json
from trl import SFTTrainer
from transformers import TrainingArguments,  DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import random
from datetime import datetime
from transformers import pipeline

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f'selection_{current_time}'
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

if True:
    model, tokenizer = FastLanguageModel.from_pretrained(
       model_name = "sft_output_query3/checkpoint-26",
       max_seq_length = max_seq_length,
       dtype = dtype,
       load_in_4bit = load_in_4bit,
       # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Phi-3.5-mini-instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
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

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-3.5",
)

max_actions = 5

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}"""

INSTRUCTIONS = "You are an expert in machine learning papers. You are given an excerpt from a paper, where a citation was deleted. I'm trying to find the citation (ignore the word [CITATION], that's just where the citation was deleted from). Read the following excerpt, and tell me what paper was cited. Tell me what to search in order to find the paper." \
f"\n\nYou can use any of the following actions up to {max_actions} times:\n" \
'Search by relevance: {"action": {"name": "search_relevance": "query": "search terms"}}\n' \
'Search by citation count: {"action": {"name": "search_citation_count": "query": "search terms"}}\n' \
'Select cited paper: {"action": {"name": "select_paper", "paper_id": "paper ID"}}\n'

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
tmpl = """{{
  "reason": "{}",
  "action": {{
    "name": "{}",
    "query": "{}"
  }}
}}"""

def format_action2(query, action, reason):
  return tmpl.format(reason, action, query)

def format_action(query, action, reason):
   return json.dumps({
        "reason": reason,
        "action": {
            "name": action,
            "query": query
        }
    })

def format_select(paper_id, reason):
    return json.dumps({
        "reason": reason,
        "action": {
            "name": "select_paper",
            "paper_id": paper_id
        }
    })

def choose_action(relevance_rank, citation_rank):
  if relevance_rank < citation_rank:
    return "search_relevance"
  else:
    return "search_citation_count"

def random_action():
  return random.choice(["search_relevance", "search_citation_count"])

def search_results(papers, relevance_rank, citation_rank):
  search_action = choose_action(relevance_rank, citation_rank)
  if search_action == 'search_citation_count':
    papers = sorted(papers, key=lambda x: x['citationCount'], reverse=True)
  papers = papers[:10]
  papers_str = ""
  for paper in papers:
      papers_str += f"- Paper ID: {paper['paperId']}\n"
      papers_str += f"\tTitle: {paper['title']}\n"
      if paper['abstract']:
          papers_str += f"\tAbstract: {paper['abstract'][:128]}\n"
      papers_str += f"\tCitation Count: {paper['citationCount']}\n\n"
  return papers_str

def formatting_prompts_func(examples):
    inputs       = examples["excerpt"]
    #outputs      = [format_action(q, choose_action(r_relevance, r_citation)) for (q, r_relevance, r_citation) in zip(examples["query"], examples["relevance_rank"], examples["citation_rank"])]
    texts = []
    for citedPaperId, input, query, reason, papers, relevance_rank, citation_rank in zip(
        examples['paperId'], inputs, examples["query"], examples['reason'], 
        examples['results'], examples["relevance_rank"], examples["citation_rank"]):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        search_action = choose_action(relevance_rank, citation_rank)
        #papers_str = search_results(papers, relevance_rank, citation_rank)
        chat = [
            { "role": "user", "content": alpaca_prompt.format(INSTRUCTIONS, input, reason) },
            { "role": "assistant", "content": '### Output:\n' +format_action(query, search_action, reason) },
            #{ "role": "user", "content": f"Here are the papers found for the given search query:\n\n{papers_str}\n\nNow select the cited paper ID using select_paper action." },
            #{ "role": "assistant", "content": '### Output:\n' +format_select(citedPaperId, selection_reason) }
        ]
        text = tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

"""
FastLanguageModel.for_inference(model)
pipe = pipeline(
      "text-generation", 
      model=model, 
      tokenizer=tokenizer,
      max_new_tokens=64,
      do_sample=False,
      return_full_text=False,
)
"""

dataset = load_from_disk("sft_dataset_20241116_215346")
 
#dataset = load_from_disk("queries_dataset_ranked_20241115_231741")
#dataset = dataset.filter(lambda x: x["relevance_rank"] <= 10 or x["citation_rank"] <= 10)
#print(len(dataset))
#dataset = dataset.map(selection_reason, batched = True, batch_size=64)
#dataset = dataset.map(formatting_prompts_func, batched = True, batch_size=256)

if False:
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 1,
            #max_steps = 50,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"sft_output_query{suffix}",
            save_steps=100,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "### Input:",
        response_part = "### Output:",
    )
    trainer.train()

"""
import gc
import time
for _ in range(5):
   time.sleep(3.0)
   torch.cuda.empty_cache()
   gc.collect()
"""

print("Training select action")

dataset_selects = Dataset.from_csv("gold_selections_20241123_115007.csv")
selected_ids = {}
for row in dataset_selects:
   selected_ids[row["paperId"]] = row

def select_prompt(item):
    k_papers = 5
    citedPaperId = item["paperId"]
    selection_reason = selected_ids[citedPaperId]["reason"]
    results = item["results"]
    target_paper = list(filter(lambda x: x['paperId'] == item['paperId'], results))
    other_papers = list(filter(lambda x: x['paperId'] != item['paperId'], results))
    random.shuffle(other_papers)
    papers = [target_paper[0]] + other_papers[:k_papers - 1] 
    # shuffle
    random.shuffle(papers)
    papers_str = ""
    for paper in papers:
        papers_str += f"- Paper ID: {paper['paperId']}\n"
        papers_str += f"\tTitle: {paper['title']}\n"
        if paper['abstract']:
            papers_str += f"\tAbstract: {paper['abstract'][:128]}\n"
        papers_str += f"\tCitation Count: {paper['citationCount']}\n\n"
    chat = [
        { "role": "user", "content": alpaca_prompt.format(INSTRUCTIONS, f"From the following academic papers list, you must select single paper:\n\n{papers_str}\n\n The reason should be single line consice short explanation of why that paper is the best match, respond with nothing but a JSON format.") },
        { "role": "assistant", "content": '### Output:\n' + format_select(citedPaperId, selection_reason) }
    ]
    return { "select":  tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt=False) }

dataset = dataset.filter(lambda x: x["paperId"] in selected_ids)
dataset = dataset.filter(lambda x: any(p['paperId'] == x['paperId'] for p in x['results']))
dataset = dataset.map(select_prompt)
print(len(dataset))
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "select",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 2,
        #max_steps = 50,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"sft_output_select{suffix}",
        save_steps=100,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Input:",
    response_part = "### Output:",
)
trainer.train()
