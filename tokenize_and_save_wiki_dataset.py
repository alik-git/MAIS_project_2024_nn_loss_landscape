from datasets import load_dataset
from transformers import BertTokenizer
import torch
from pathlib import Path
import logging
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load dataset
logging.info("Loading the Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)['train']

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Directory to save tokenized data
save_dir = Path("./tokenized_data")
save_dir.mkdir(parents=True, exist_ok=True)

# Process and save the dataset in chunks
chunk_size = 1000  # Define the size of each chunk
total_size = len(dataset)

# Setting up tqdm progress bar
progress_bar = tqdm(range(0, total_size, chunk_size), desc="Processing chunks")

for i in progress_bar:
    # Select a subset of data to process
    subset = dataset.select(range(i, min(i + chunk_size, total_size)))

    # Tokenize data using multiple processes
    tokenized_data = subset.map(tokenize_function, batched=True, num_proc=4)

    # Save the tokenized data to disk
    save_path = save_dir / f"tokenized_data_{i}.pt"
    torch.save(tokenized_data, save_path)
    logging.info(f"Saved tokenized data to {save_path}")
    progress_bar.set_description(f"Processing and saving chunk starting at index {i}")

logging.info("Tokenization and saving completed.")
