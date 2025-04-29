from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load Legal-BERT (Dutch & English) embedding pipeline
def load_legalbert_embedding_pipeline():
    model_name = "Gerwin/legal-bert-dutch-english"
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def embed(text: str) -> list[float]:
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        # Forward pass without gradient computation
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean-pooling of the last hidden state
        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch_size, seq_len, 1]
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts
        # Convert to Python list
        return mean_pooled.squeeze().cpu().numpy().tolist()

    return embed
