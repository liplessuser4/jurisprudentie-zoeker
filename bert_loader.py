from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    pipeline
)
import torch

def load_legalbert_embedding_pipeline():
    model_name = "Gerwin/legal-bert-dutch-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def embed(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return embed


def load_ner_pipeline():
    model_name = "Davlan/xlm-roberta-base-finetuned-conll03-dutch"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
