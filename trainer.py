#!/usr/bin/env python3
"""
Electronics LLM Trainer - Core Training Module
Memory-aware text extraction and training optimized for electronics textbooks
Intelligent resource management with 75% memory threshold protection

Copyright (c) 2025 SysQuirrel
Licensed under the MIT License - see LICENSE file for details
"""

import os
import sys
import logging
import torch
import json
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import PyPDF2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
BOOKS_DIR = Path("books")
OUTPUT_DIR = Path("trained_model")
CACHE_DIR = Path("cache")
PROCESSED_DIR = Path("processed")

# Create directories
for dir_path in [OUTPUT_DIR, CACHE_DIR, PROCESSED_DIR]:
    dir_path.mkdir(exist_ok=True)

class SimpleLLMTrainer:
    """Simplified LLM trainer focusing on text extraction only"""
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"  # 345M params for better quality
        self.max_length = 256  # Longer sequences for better context
        self.batch_size = 2   # Smaller batch for larger model
        self.learning_rate = 1e-4  # More conservative learning rate
        self.epochs = 1

    def extract_text_from_pdf(self, pdf_path):
        """Memory-aware text extraction from PDF with 75% resource threshold"""
        import psutil
        import gc
        
        logger.info(f"Extracting text from {pdf_path.name}...")
        text_content = ""
        
        # Get system memory info
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        memory_threshold = 0.75  # Use 75% of available memory
        max_memory_mb = (available_memory_gb * memory_threshold * 1024)
        
        logger.info(f"Available memory: {available_memory_gb:.1f}GB, using max {max_memory_mb:.0f}MB")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Found {total_pages} pages")
                
                processed_pages = 0
                current_memory_mb = 0
                
                for page_num in range(total_pages):
                    try:
                        # Check memory usage before processing each page
                        process = psutil.Process()
                        current_memory_mb = process.memory_info().rss / (1024**2)
                        
                        # Stop if we're approaching memory limit
                        if current_memory_mb > max_memory_mb:
                            logger.warning(f"Memory threshold reached ({current_memory_mb:.0f}MB > {max_memory_mb:.0f}MB)")
                            logger.info(f"Processed {processed_pages}/{total_pages} pages ({processed_pages/total_pages*100:.1f}%) before hitting memory limit")
                            break
                        
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"\n\nPage {page_num + 1}:\n{page_text}"
                            processed_pages += 1
                            
                        # Log progress every 25 pages and show memory usage
                        if (page_num + 1) % 25 == 0:
                            logger.info(f"Processed {processed_pages}/{total_pages} pages, Memory: {current_memory_mb:.0f}MB")
                            
                        # Garbage collect every 50 pages to manage memory
                        if (page_num + 1) % 50 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                # Final memory cleanup
                gc.collect()
                        
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
            
        final_words = len(text_content.split())
        logger.info(f"Extracted {final_words} words from {processed_pages}/{total_pages} pages of {pdf_path.name}")
        logger.info(f"Coverage: {processed_pages/total_pages*100:.1f}% of the book")
        return text_content

    def process_books(self):
        """Process books with simple text extraction only"""
        pdf_files = list(BOOKS_DIR.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF books for processing")
        
        all_content = []
        processed_books_info = []
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing book {i+1}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Get original file size
                original_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
                
                # Extract text content only
                content = self.extract_text_from_pdf(pdf_file)
                
                if content and len(content.strip()) > 100:  # At least 100 characters
                    # Filter for relevant content (if needed)
                    content_lower = content.lower()
                    
                    # Save extracted content
                    content_file = PROCESSED_DIR / f"{pdf_file.stem}_extracted.txt"
                    with open(content_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Extracted from: {pdf_file.name}\n")
                        f.write(f"# Original size: {original_size:.2f} MB\n")
                        f.write(f"# Processing date: {datetime.now().isoformat()}\n")
                        f.write("# Content: Text extraction only\n\n")
                        f.write(content)
                    
                    all_content.append(content)
                    word_count = len(content.split())
                    content_size = len(content.encode('utf-8')) / 1024  # KB
                    
                    # Delete original PDF to save space
                    pdf_file.unlink()
                    
                    processed_books_info.append({
                        'name': pdf_file.name,
                        'original_size_mb': original_size,
                        'extracted_size_kb': content_size,
                        'word_count': word_count,
                        'space_saved_mb': original_size - (content_size / 1024)
                    })
                    
                    logger.info(f"  ✅ Successfully processed {pdf_file.name}")
                    logger.info(f"     📊 {word_count} words extracted")
                    logger.info(f"     💾 Saved {original_size - (content_size / 1024):.2f} MB space")
                else:
                    logger.warning(f"  ⏭️ Skipped {pdf_file.name} (insufficient content)")
                    
            except Exception as e:
                logger.error(f"  💥 Error processing {pdf_file.name}: {e}")
                continue
        
        # Save processing summary
        if processed_books_info:
            summary_file = PROCESSED_DIR / "processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'processed_books': processed_books_info,
                    'total_books': len(processed_books_info),
                    'total_space_saved_mb': sum(book['space_saved_mb'] for book in processed_books_info),
                    'total_words': sum(book['word_count'] for book in processed_books_info)
                }, f, indent=2)
            
            total_saved = sum(book['space_saved_mb'] for book in processed_books_info)
            logger.info(f"📈 PROCESSING SUMMARY:")
            logger.info(f"   📚 Processed: {len(processed_books_info)} books")
            logger.info(f"   💾 Space saved: {total_saved:.2f} MB")
            logger.info(f"   📄 Content saved to: {PROCESSED_DIR}")
        
        logger.info(f"Successfully processed {len(all_content)} books")
        return all_content

    def prepare_dataset(self, texts):
        """Prepare training and evaluation datasets"""
        logger.info("Preparing datasets...")
        
        # Split each text into smaller chunks
        all_chunks = []
        for text in texts:
            # Split by paragraphs and sentences
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if len(para.strip()) > 50:  # At least 50 characters
                    # Further split long paragraphs
                    sentences = para.split('. ')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 400:  # Keep chunks reasonable
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                all_chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    if current_chunk.strip():
                        all_chunks.append(current_chunk.strip())
        
        # Remove duplicates and very short chunks
        unique_chunks = list(set([chunk for chunk in all_chunks if len(chunk) > 20]))
        logger.info(f"Created {len(unique_chunks)} unique text chunks")
        
        # Split for training and evaluation
        split_idx = int(len(unique_chunks) * 0.9)
        train_texts = unique_chunks[:split_idx]
        eval_texts = unique_chunks[split_idx:]
        
        logger.info(f"Training chunks: {len(train_texts)}")
        logger.info(f"Evaluation chunks: {len(eval_texts)}")
        
        return train_texts, eval_texts

    def train(self):
        """Main training function with error handling"""
        try:
            logger.info("=== STARTING SIMPLE LLM TRAINING ===")
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Working directory: {os.getcwd()}")
            
            # Check disk space
            import shutil
            total, used, free = shutil.disk_usage('.')
            logger.info(f"Available disk space: {free / (1024**3):.2f} GB")
            
            # Step 1: Process books (text only)
            logger.info("Step 1: Processing PDF books (text extraction only)...")
            all_texts = self.process_books()
            
            if not all_texts:
                logger.error("No text extracted from books!")
                return False
            
            logger.info(f"Successfully extracted content from {len(all_texts)} books")
            
            # Step 2: Prepare dataset
            logger.info("Step 2: Preparing training dataset...")
            train_texts, eval_texts = self.prepare_dataset(all_texts)
            
            if not train_texts:
                logger.error("No training data prepared!")
                return False
            
            # Step 3: Create datasets
            logger.info("Step 3: Creating datasets...")
            train_dataset = Dataset.from_dict({"text": train_texts})
            eval_dataset = Dataset.from_dict({"text": eval_texts})
            
            # Step 4: Load model and tokenizer
            logger.info(f"Step 4: Loading model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(CACHE_DIR),
                trust_remote_code=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(CACHE_DIR),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Resize embeddings if needed
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Model and tokenizer loaded successfully!")
            
            # Step 5: Tokenize datasets
            logger.info("Step 5: Tokenizing datasets...")
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
                mlm=False
            )
            
            # Step 6: Training arguments
            logger.info("Step 6: Setting up training...")
            training_args = TrainingArguments(
                output_dir=str(OUTPUT_DIR),
                overwrite_output_dir=True,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                eval_strategy="steps",
                save_strategy="steps", 
                save_steps=100,  # More frequent saves
                eval_steps=100,  # More frequent evaluation
                logging_steps=10,  # Detailed logging for monitoring
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                warmup_steps=100,  # More warmup for stability
                load_best_model_at_end=True,
                gradient_accumulation_steps=8,  # Larger effective batch size
                fp16=False,  # Disable mixed precision for CPU
                dataloader_pin_memory=False,
                report_to=[],
                logging_first_step=True,
            )
            
            # Step 7: Create trainer
            logger.info("Step 7: Initializing trainer...")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Step 8: Train the model
            logger.info("Step 8: Starting training...")
            trainer.train()
            
            # Step 9: Save the model
            logger.info("Step 9: Saving trained model...")
            trainer.save_model(str(OUTPUT_DIR / "final_model"))
            tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
            
            logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

# Run the trainer when executed directly
if __name__ == "__main__":
    try:
        logger.info("Starting Simple LLM Trainer...")
        trainer = SimpleLLMTrainer()
        success = trainer.train()
        
        if success:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)