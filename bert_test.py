from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load dataset
logging.info("Loading the Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)['train']
subset_dataset = dataset.select(range(10))  # Using a subset for quick processing

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Function to tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')

# Tokenize dataset
logging.info("Tokenizing data...")
tokenized_datasets = subset_dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# DataLoader
logging.info("Setting up DataLoader...")
train_dataloader = DataLoader(tokenized_datasets, batch_size=16, collate_fn=data_collator)

# Training model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

logging.info("Starting training loop...")
try:
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logging.info(f"Batch loss: {loss.item()}")

        # Compute the gradients of the loss
        model.zero_grad()  # Reset gradients to zero to avoid accumulation
        loss.backward()    # Backpropagation to compute gradients

        # Print gradients of each parameter
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                logging.info(f"Gradient of {name}: {parameter.grad}")
        
        break  # Break after first batch for demonstration
except Exception as e:
    logging.error(f"An error occurred: {e}")

logging.info("Training loop completed.")
k=1