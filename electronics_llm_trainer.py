#!/usr/bin/env python3
"""
Electronics LLM Trainer for Daytona
This script runs in the Daytona sandbox to train an LLM on electronics textbooks.
"""

import os
import sys
import logging
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import PyPDF2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BOOKS_DIR = Path("books")
OUTPUT_DIR = Path("trained_model")
CACHE_DIR = Path("cache")
PROCESSED_DIR = Path("processed")

class ElectronicsLLMTrainer:
    """Train an LLM on electronics textbooks"""
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"  # Starting point for training
        self.max_length = 512
        self.batch_size = 4
        self.learning_rate = 5e-5
        self.epochs = 3
        
        # Create necessary directories
        for directory in [BOOKS_DIR, OUTPUT_DIR, CACHE_DIR, PROCESSED_DIR]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return ""
    
    def process_books(self):
        """Process all PDF books in the books directory"""
        pdf_files = list(BOOKS_DIR.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF books")
        
        all_text = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                # Only keep sections with electronics-related content
                if any(term in text.lower() for term in 
                      ["circuit", "electronics", "voltage", "current", 
                       "resistor", "capacitor", "transistor"]):
                    all_text.append(text)
                    logger.info(f"  Added {len(text.split())} words from {pdf_file.name}")
        
        return all_text
    
    def create_dataset(self, texts):
        """Create dataset from extracted texts"""
        # Split texts into smaller chunks for training
        chunks = []
        for text in texts:
            # Split into paragraphs
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                if len(para.split()) > 20:  # Only use paragraphs with at least 20 words
                    chunks.append(para)
        
        logger.info(f"Created {len(chunks)} text chunks for training")
        
        # Create a Hugging Face dataset
        dataset = Dataset.from_dict({"text": chunks})
        return dataset
    
    def train(self):
        """Train the model on the dataset"""
        # Process books
        texts = self.process_books()
        if not texts:
            logger.error("No text extracted from books. Please add PDF books to the 'books' directory.")
            return
        
        # Create dataset
        dataset = self.create_dataset(texts)
        
        # Split dataset
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        # Load tokenizer and model
        logger.info(f"Loading model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Not masked language modeling
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the trained model
        logger.info("Saving trained model...")
        trainer.save_model(str(OUTPUT_DIR / "final_model"))
        tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
        
        logger.info("Training complete!")

# Run the trainer when executed directly
if __name__ == "__main__":
    logger.info("Starting Electronics LLM Trainer...")
    trainer = ElectronicsLLMTrainer()
    trainer.train()