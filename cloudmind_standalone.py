#!/usr/bin/env python3
"""
Daytona Electronics LLM Trainer
Deploy and train language models on electronics textbooks using Daytona cloud sandboxes
Consumes Daytona credits efficiently for maximum computational value
"""

import os
import logging
from pathlib import Path
from daytona import Daytona, DaytonaConfig, Resources

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daytona_llm_trainer.log'),
        logging.StreamHandler()
    ]
)

class DaytonaLLMTrainer:
    """Deploy and train an LLM on Daytona's cloud sandbox"""

    def __init__(self, api_key: str):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.daytona = Daytona(DaytonaConfig(api_key=self.api_key))
        self.sandbox = None

    def create_sandbox(self):
        """Create a Daytona sandbox for training with high-performance resources"""
        self.logger.info("Creating Daytona sandbox with 10 vCPU, 10GB RAM, 30GB disk...")
        
        # Create the sandbox with specified resources
        self.sandbox = self.daytona.create(
            cpu=10,    # 10 CPU cores
            memory=10, # 10GB RAM
            disk=30    # 30GB disk space
        )
        self.logger.info(f"Sandbox created with ID: {self.sandbox.id}")

    def upload_files(self):
        """Upload training script and books to the sandbox"""
        # Upload the training script
        self.logger.info("Uploading training script to sandbox...")
        with open("electronics_llm_trainer.py", 'rb') as script_file:
            self.sandbox.fs.upload_file(script_file.read(), "electronics_llm_trainer.py")
        
        # Upload requirements
        with open("requirements.txt", 'rb') as req_file:
            self.sandbox.fs.upload_file(req_file.read(), "requirements.txt")
        
        # Create books directory in the sandbox
        self.sandbox.process.code_run("mkdir -p books")
        
        # Upload books if they exist
        books_dir = Path("books")
        if books_dir.exists():
            pdf_files = list(books_dir.glob("*.pdf"))
            self.logger.info(f"Found {len(pdf_files)} PDF books to upload")
            
            for pdf_file in pdf_files:
                self.logger.info(f"Uploading {pdf_file.name}...")
                with open(pdf_file, 'rb') as f:
                    self.sandbox.fs.upload_file(f.read(), f"books/{pdf_file.name}")
        else:
            self.logger.warning("No books directory found. Will use sample data.")
        
        self.logger.info("All files uploaded successfully.")

    def install_dependencies(self):
        """Install required packages in the sandbox"""
        self.logger.info("Installing dependencies in sandbox...")
        response = self.sandbox.process.code_run("pip install -r requirements.txt")
        if response.exit_code != 0:
            self.logger.error(f"Failed to install dependencies: {response.result}")
            return False
        self.logger.info("Dependencies installed successfully.")
        return True

    def run_training(self):
        """Run the training script inside the sandbox"""
        self.logger.info("Starting training inside the sandbox...")
        response = self.sandbox.process.code_run("python electronics_llm_trainer.py")
        if response.exit_code != 0:
            self.logger.error(f"Training failed: {response.result}")
        else:
            self.logger.info("Training completed successfully.")
            self.logger.info(response.result)

    def download_model(self):
        """Download the trained model from the sandbox"""
        self.logger.info("Downloading trained model...")
        try:
            # Create local directory for the model
            os.makedirs("trained_model", exist_ok=True)
            
            # List files in the trained_model directory in the sandbox
            response = self.sandbox.process.code_run("ls -la trained_model/final_model")
            if response.exit_code == 0:
                # Download each file
                model_files = response.result.split("\n")
                for file_line in model_files:
                    if file_line.startswith("d") or not file_line.strip():
                        continue  # Skip directories and empty lines
                    
                    # Extract filename
                    parts = file_line.split()
                    if len(parts) >= 9:
                        filename = " ".join(parts[8:])
                        self.logger.info(f"Downloading {filename}...")
                        
                        # Download the file
                        file_content = self.sandbox.fs.download_file(f"trained_model/final_model/{filename}")
                        with open(f"trained_model/{filename}", "wb") as f:
                            f.write(file_content)
                
                self.logger.info("Model downloaded successfully!")
            else:
                self.logger.error("Failed to list model files. Model may not have been saved.")
        except Exception as e:
            self.logger.error(f"Error downloading model: {str(e)}")

    def cleanup(self):
        """Delete the sandbox to free resources"""
        if self.sandbox:
            self.logger.info("Cleaning up sandbox...")
            self.sandbox.delete()
            self.logger.info("Sandbox deleted.")

if __name__ == "__main__":
    API_KEY = "dtn_eaa634a793b76c1aa5f949f9646e665dd8989e2ac0280b9add1177fb0a58ce1f"  # Replace with your actual API key
    
    trainer = DaytonaLLMTrainer(api_key=API_KEY)
    try:
        trainer.create_sandbox()
        trainer.upload_files()
        if trainer.install_dependencies():
            trainer.run_training()
            trainer.download_model()
    finally:
        trainer.cleanup()
    
    trainer = DaytonaLLMTrainer(api_key=API_KEY)
    try:
        trainer.create_sandbox()
        trainer.upload_files()
        if trainer.install_dependencies():
            trainer.run_training()
            trainer.download_model()
    finally:
        trainer.cleanup()