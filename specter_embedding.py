from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Interface with embed([text]) -> tensor
class EmbeddingWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_parallel(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512, return_attention_mask=True).to(self.model.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    
    def embedding_model(self):
        raise NotImplementedError()
    
    def embedding_dim(self):
        raise NotImplementedError()
    
class Specter2AdhocQuery(EmbeddingWrapper):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
        model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="specter2_adhoc_query", set_active=True)

        super().__init__(model, tokenizer)

    def embedding_model(self):
        return "specter2"

    def embedding_dim(self):
        return 768
    
class Specter2Document(EmbeddingWrapper):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
        model.load_adapter("allenai/specter2", source="hf", load_as="specter2_proximity", set_active=True)
        super().__init__(model, tokenizer)
    
    def embedding_model(self):
        return "specter2"
    
    def embedding_dim(self):
        return 768