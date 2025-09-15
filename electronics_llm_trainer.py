#!/usr/bin/env python3
"""
Electronics LLM Trainer for Daytona
This script runs in the Daytona sandbox to train an LLM on electronics textbooks.
"""

import os
import sys
import logging
import torch
import cv2
import numpy as np
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
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import re

# Set environment variables for better download and caching
os.environ["HF_HOME"] = str(Path("cache"))  # Set Hugging Face cache directory
os.environ["TRANSFORMERS_CACHE"] = str(Path("cache"))  # Set transformers cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings in sandbox

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
        # Use a smaller but capable model that fits in 10GB disk space
        self.model_name = "microsoft/DialoGPT-medium"  # 1.2GB model - fits comfortably
        # Alternative smaller options:
        # "distilgpt2" # 319MB
        # "gpt2" # 548MB  
        # "microsoft/DialoGPT-small" # 387MB
        
        # Configuration optimized for CPU training within resource limits
        self.max_length = 256  # Reduced for CPU training
        self.batch_size = 2    # Small batch size for CPU
        self.learning_rate = 5e-5
        self.epochs = 1        # Single epoch due to time constraints
        
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

    def analyze_electronics_image(self, image_array):
        """Analyze image for electronics components using computer vision"""
        try:
            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            components_found = []
            
            # Detect circles (often capacitors, inductors)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=50)
            if circles is not None:
                components_found.append(f"{len(circles[0])} circular components")
            
            # Detect rectangles (ICs, resistors)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4 and cv2.contourArea(contour) > 100:
                    rectangles += 1
            if rectangles > 0:
                components_found.append(f"{rectangles} rectangular components")
            
            # Detect lines (wires, connections)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)
            if lines is not None:
                components_found.append(f"{len(lines)} connection lines")
            
            if components_found:
                return f"Circuit diagram with: {', '.join(components_found)}"
            else:
                return "Electronic schematic or diagram"
                
        except Exception as e:
            logger.warning(f"Error analyzing image: {e}")
            return "Electronic diagram"

    def extract_ocr_with_preprocessing(self, image):
        """Extract text from image with preprocessing for better OCR"""
        try:
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increase contrast
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)  # Increase sharpness
            
            # OCR with different configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 8',  # Single word
                '--psm 13', # Raw line (for equations)
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                except:
                    continue
                    
            return best_text.strip()
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return ""

    def extract_multimodal_content(self, pdf_path):
        """Extract both text and image content from PDF"""
        logger.info(f"Processing multimodal content from {pdf_path.name}...")
        
        # 1. Extract regular text
        text_content = self.extract_text_from_pdf(pdf_path)
        
        # 2. Extract image content
        image_descriptions = []
        ocr_texts = []
        
        try:
            # Convert PDF to images with low DPI to save memory
            logger.info("Converting PDF pages to images...")
            pages = convert_from_path(pdf_path, dpi=100, first_page=1, last_page=10)  # Limit to first 10 pages
            
            for page_num, page_image in enumerate(pages):
                logger.info(f"Processing page {page_num + 1}...")
                
                # OCR for text in images (equations, labels, component values)
                ocr_text = self.extract_ocr_with_preprocessing(page_image)
                
                # Look for electronics-specific content in OCR
                if any(term in ocr_text.lower() for term in 
                      ['ω', 'µ', 'voltage', 'current', 'resistance', 'capacitance',
                       'v =', 'i =', 'r =', 'c =', 'l =', 'f =', 'khz', 'mhz']):
                    ocr_texts.append(f"Page {page_num + 1} technical text: {ocr_text}")
                
                # Analyze image for circuit components
                if len(ocr_text.strip()) < 50:  # If little text, likely a diagram
                    circuit_desc = self.analyze_electronics_image(page_image)
                    image_descriptions.append(f"Page {page_num + 1}: {circuit_desc}")
                
                # Free memory immediately
                del page_image
            
        except Exception as e:
            logger.warning(f"Error processing images from {pdf_path}: {e}")
        
        # Combine all content
        all_content = [text_content]
        if ocr_texts:
            all_content.append("OCR EXTRACTED CONTENT:\n" + "\n".join(ocr_texts))
        if image_descriptions:
            all_content.append("IMAGE ANALYSIS:\n" + "\n".join(image_descriptions))
        
        combined_content = "\n\n".join(filter(None, all_content))
        
        logger.info(f"Extracted {len(text_content.split())} words from text, "
                   f"{len(' '.join(ocr_texts).split())} words from OCR, "
                   f"{len(image_descriptions)} image descriptions")
        
        return combined_content
    
    def process_books(self):
        """Process all PDF books with multimodal content extraction and space optimization"""
        pdf_files = list(BOOKS_DIR.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF books for multimodal processing")
        
        all_content = []
        processed_books_info = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name} with multimodal extraction...")
            
            try:
                # Get original file size for logging
                original_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
                
                # Use multimodal extraction instead of text-only
                content = self.extract_multimodal_content(pdf_file)
                
                if content:
                    # Check for electronics-related content in combined text
                    content_lower = content.lower()
                    if any(term in content_lower for term in 
                          ["circuit", "electronics", "voltage", "current", 
                           "resistor", "capacitor", "transistor", "schematic",
                           "component", "amplifier", "frequency", "impedance"]):
                        
                        # Save extracted content to individual file
                        content_file = PROCESSED_DIR / f"{pdf_file.stem}_extracted.txt"
                        with open(content_file, 'w', encoding='utf-8') as f:
                            f.write(f"# Extracted from: {pdf_file.name}\n")
                            f.write(f"# Original size: {original_size:.2f} MB\n")
                            f.write(f"# Processing date: {datetime.now().isoformat()}\n")
                            f.write("# Content includes: Text + OCR + Image analysis\n\n")
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
                        
                        logger.info(f"  ✅ Processed {pdf_file.name}:")
                        logger.info(f"     📊 {word_count} words extracted")
                        logger.info(f"     💾 Saved {original_size - (content_size / 1024):.2f} MB space")
                        logger.info(f"     🗑️ Original PDF deleted")
                    else:
                        logger.info(f"  ⏭️ Skipped {pdf_file.name} (no electronics content)")
                else:
                    logger.warning(f"  ❌ No content extracted from {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"  💥 Error processing {pdf_file.name}: {e}")
                continue
        
        # Save processing summary
        if processed_books_info:
            summary_file = PROCESSED_DIR / "processing_summary.json"
            import json
            with open(summary_file, 'w') as f:
                json.dump({
                    'processed_books': processed_books_info,
                    'total_books': len(processed_books_info),
                    'total_space_saved_mb': sum(book['space_saved_mb'] for book in processed_books_info),
                    'total_words': sum(book['word_count'] for book in processed_books_info)
                }, f, indent=2)
            
            total_saved = sum(book['space_saved_mb'] for book in processed_books_info)
            logger.info(f"📈 SPACE OPTIMIZATION SUMMARY:")
            logger.info(f"   📚 Processed: {len(processed_books_info)} books")
            logger.info(f"   💾 Space saved: {total_saved:.2f} MB")
            logger.info(f"   📄 Content saved to: {PROCESSED_DIR}")
        
        logger.info(f"Successfully processed {len(all_content)} books with multimodal extraction")
        return all_content
    
    def create_dataset(self, texts):
        """Create dataset from extracted texts with improved processing for technical content"""
        # Split texts into chunks suitable for technical training
        chunks = []
        for text in texts:
            # Split into sections (chapters, subsections, etc.)
            sections = text.split("\n\n")
            
            current_chunk = ""
            for section in sections:
                # Filter out very short sections and page numbers
                if len(section.split()) < 10:
                    continue
                    
                # Combine sections into larger chunks
                if len((current_chunk + " " + section).split()) <= 400:  # Increased chunk size
                    current_chunk += " " + section if current_chunk else section
                else:
                    if current_chunk and len(current_chunk.split()) > 30:
                        # Add instruction format for better learning
                        formatted_chunk = f"Electronics Knowledge: {current_chunk.strip()}"
                        chunks.append(formatted_chunk)
                    current_chunk = section
            
            # Add the last chunk
            if current_chunk and len(current_chunk.split()) > 30:
                formatted_chunk = f"Electronics Knowledge: {current_chunk.strip()}"
                chunks.append(formatted_chunk)
        
        logger.info(f"Created {len(chunks)} technical text chunks for training")
        
        # Create a Hugging Face dataset
        dataset = Dataset.from_dict({"text": chunks})
        return dataset
    
    def train(self):
        """Train the model on the dataset"""
        logger.info("Starting Electronics LLM Training...")
        logger.info("=== TRAINING CONFIGURATION ===")
        logger.info(f"Base Model: {self.model_name}")
        logger.info(f"Max Length: {self.max_length}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info("===============================")
        
        # Process books
        logger.info("Step 1: Processing PDF books...")
        texts = self.process_books()
        if not texts:
            logger.error("No text extracted from books. Please add PDF books to the 'books' directory.")
            return
        
        # Create dataset
        logger.info("Step 2: Creating training dataset...")
        dataset = self.create_dataset(texts)
        
        # Split dataset
        logger.info("Step 3: Splitting dataset...")
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # Load tokenizer and model (will download automatically in sandbox)
        logger.info(f"Loading model: {self.model_name}")
        logger.info("Model will be downloaded automatically from Hugging Face...")
        
        # Download tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR)  # Cache in our designated directory
        )
        
        # Download and load model with CPU-only optimizations
        logger.info("Downloading and loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),  # Cache in our designated directory
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        )
        
        logger.info("Model loaded successfully!")
        
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Resize model embeddings if needed
        model.resize_token_embeddings(len(tokenizer))
        
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
        
        # Training arguments optimized for model fine-tuning
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=250,  # More frequent saves
            eval_steps=250,
            logging_steps=50,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=100,  # Fewer warmup steps for fine-tuning
            load_best_model_at_end=True,
            gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
            fp16=True,  # Use mixed precision for memory efficiency
            dataloader_pin_memory=False,  # Reduce memory usage
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
    try:
        logger.info("Starting Electronics LLM Trainer...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check available disk space using shutil
        import shutil
        total, used, free = shutil.disk_usage('.')
        logger.info(f"Available disk space: {free / (1024**3):.2f} GB")
        
        trainer = ElectronicsLLMTrainer()
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise