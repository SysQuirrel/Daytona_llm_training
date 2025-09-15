#!/usr/bin/env python3
"""
Daytona         # Create resources object with custom specifications (within account limits)
        resources = Resources(
            cpu=4,      # 4 CPU cores (maximum allowed)
            memory=6,   # 6GB RAM (testing limit)
            disk=10     # 10GB disk space (maximum allowed)
        )onics LLM Trainer
Deploy and train language models on electronics textbooks using Daytona cloud sandboxes
Consumes Daytona credits efficiently for maximum computational value
"""

import os
import logging
from pathlib import Path
from daytona import Daytona, DaytonaConfig, Resources, CreateSandboxFromImageParams, Image

# Setup comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all log levels
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('daytona_llm_trainer.log', mode='w'),  # Overwrite each run
        logging.StreamHandler()
    ]
)

# Also log to a separate error file
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler('daytona_errors.log', mode='w')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - ERROR - %(message)s'))
error_logger.addHandler(error_handler)

class DaytonaLLMTrainer:
    """Deploy and train an LLM on Daytona's cloud sandbox"""

    def __init__(self, api_key: str):
        self.logger = logging.getLogger(__name__)
        self.error_logger = logging.getLogger('errors')
        self.api_key = api_key
        self.daytona = Daytona(DaytonaConfig(api_key=self.api_key))
        self.sandbox = None

    def log_operation(self, operation_name: str, details: dict = None, error: Exception = None):
        """Log detailed information about operations for debugging"""
        self.logger.info(f"=== {operation_name} ===")
        if details:
            for key, value in details.items():
                self.logger.info(f"{key}: {value}")
        if error:
            self.logger.error(f"ERROR in {operation_name}: {str(error)}")
            self.error_logger.error(f"{operation_name}: {str(error)}")
        self.logger.info(f"=== END {operation_name} ===")

    def log_sandbox_response(self, response, operation: str):
        """Log detailed sandbox response information"""
        self.logger.debug(f"--- {operation} Response ---")
        if hasattr(response, 'exit_code'):
            self.logger.debug(f"Exit Code: {response.exit_code}")
        if hasattr(response, 'result'):
            self.logger.debug(f"Result: {response.result}")
        if hasattr(response, 'stdout'):
            self.logger.debug(f"STDOUT: {response.stdout}")
        if hasattr(response, 'stderr'):
            self.logger.debug(f"STDERR: {response.stderr}")
        self.logger.debug(f"--- END {operation} Response ---")

    def create_sandbox(self):
        """Create a Daytona sandbox for training with high-performance resources"""
        try:
            self.log_operation("SANDBOX_CREATION_START", {
                "target_cpu": 4,
                "target_memory": "4GB", 
                "target_disk": "10GB",
                "base_image": "python:3.12-slim"
            })
            
            # Create resources object with custom specifications (within account limits)
            resources = Resources(
                cpu=4,      # 4 CPU cores (conservative)
                memory=8,   # 4GB RAM (fits within 10GB total quota)
                disk=10      # 8GB disk space (conservative)
            )
            
            self.logger.info("Building image with pre-installed packages...")
            # Create sandbox parameters with custom resources - using Debian slim for PyTorch compatibility
            # Alpine has glibc/musl compatibility issues with PyTorch
            image = (Image.base("python:3.12-slim")
                    .run_commands("apt-get update && apt-get install -y tesseract-ocr poppler-utils libopencv-dev")
                    .pip_install("torch", "torchvision", 
                               index_url="https://download.pytorch.org/whl/cpu")
                    .pip_install("transformers>=4.35.0", "datasets>=2.12.0", 
                               "PyPDF2>=3.0.0", "numpy>=1.24.0", "tqdm>=4.65.0")
                    .pip_install("pytesseract>=0.3.10", "pdf2image>=1.16.0", 
                               "Pillow>=10.0.0", "opencv-python-headless>=4.8.0"))
            
            params = CreateSandboxFromImageParams(
                image=image,
                resources=resources
            )
            
            self.logger.info("Requesting sandbox creation from Daytona API...")
            # Create the sandbox
            self.sandbox = self.daytona.create(params)
            
            self.log_operation("SANDBOX_CREATION_SUCCESS", {
                "sandbox_id": self.sandbox.id,
                "status": getattr(self.sandbox, 'status', 'unknown')
            })
            
        except Exception as e:
            self.log_operation("SANDBOX_CREATION_FAILED", error=e)
            raise

    def upload_essential_files(self):
        """Upload only essential files needed for dependency installation"""
        # Upload the training script
        self.logger.info("Uploading essential files to sandbox...")
        with open("electronics_llm_trainer.py", 'rb') as script_file:
            self.sandbox.fs.upload_file(script_file.read(), "electronics_llm_trainer.py")
        
        # Upload requirements
        with open("requirements.txt", 'rb') as req_file:
            self.sandbox.fs.upload_file(req_file.read(), "requirements.txt")
        
        self.logger.info("Essential files uploaded successfully.")

    def upload_books(self):
        """Upload PDF books to the sandbox (after dependencies are installed)"""
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
        
        self.logger.info("All books uploaded successfully.")

    def install_dependencies(self):
        """Check that packages are available in the sandbox (pre-installed in image)"""
        try:
            self.log_operation("DEPENDENCY_VERIFICATION_START")
            
            # Quick verification that packages are available
            verify_command = """
import sys
import traceback
try:
    print("=== PACKAGE VERIFICATION START ===")
    
    print("Importing torch...")
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    print("Importing transformers...")
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
    
    print("Importing datasets...")
    import datasets
    print(f"✓ Datasets version: {datasets.__version__}")
    
    print("Importing PyPDF2...")
    import PyPDF2
    print(f"✓ PyPDF2 available")
    
    print("Importing numpy...")
    import numpy
    print(f"✓ NumPy version: {numpy.__version__}")
    
    print("Importing tqdm...")
    import tqdm
    print(f"✓ tqdm version: {tqdm.__version__}")
    
    print("Importing OCR and image processing packages...")
    import pytesseract
    print(f"✓ pytesseract available")
    
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
    
    from pdf2image import convert_from_path
    print(f"✓ pdf2image available")
    
    from PIL import Image
    print(f"✓ Pillow available")
    
    # Test tesseract binary
    try:
        result = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR version: {result}")
    except Exception as e:
        print(f"⚠️ Tesseract warning: {e}")
    
    print("=== ALL PACKAGES VERIFIED SUCCESSFULLY (INCLUDING MULTIMODAL) ===")
    
except Exception as e:
    print(f"=== PACKAGE VERIFICATION FAILED ===")
    print(f"ERROR: {str(e)}")
    print("TRACEBACK:")
    traceback.print_exc()
    sys.exit(1)
"""
            
            self.logger.info("Running package verification script...")
            response = self.sandbox.process.code_run(verify_command)
            
            self.log_sandbox_response(response, "PACKAGE_VERIFICATION")
            
            if response.exit_code != 0:
                self.log_operation("DEPENDENCY_VERIFICATION_FAILED", {
                    "exit_code": response.exit_code,
                    "output": response.result
                })
                return False
            
            self.log_operation("DEPENDENCY_VERIFICATION_SUCCESS", {
                "verification_output": response.result
            })
            return True
            
        except Exception as e:
            self.log_operation("DEPENDENCY_VERIFICATION_ERROR", error=e)
            return False

    def run_training(self):
        """Run the training script inside the sandbox as a background process"""
        import time
        
        try:
            self.log_operation("TRAINING_START")
            
            # Start training as a background process to avoid gateway timeouts
            self.logger.info("Starting training script in background...")
            start_command = """
import subprocess
import sys
import os
import time

# Ensure we're in the right directory
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    # Start training in background and redirect output to log file
    with open('training_output.log', 'w') as log_file:
        # Add unbuffered output to see real-time logs
        process = subprocess.Popen([sys.executable, '-u', 'electronics_llm_trainer.py'], 
                                  stdout=log_file, 
                                  stderr=subprocess.STDOUT,
                                  cwd=os.getcwd(),
                                  bufsize=0)  # Unbuffered
        
        # Write process ID to file for monitoring
        with open('training_pid.txt', 'w') as pid_file:
            pid_file.write(str(process.pid))
        
        print(f"Training started with PID: {process.pid}")
        print("Check training_output.log for progress")
        
        # Let it run for a moment to capture any immediate errors
        time.sleep(5)
        
        # Check if process is still running
        poll_result = process.poll()
        if poll_result is not None:
            print(f"WARNING: Process exited quickly with code: {poll_result}")
            # Read any immediate output
            try:
                with open('training_output.log', 'r') as f:
                    output = f.read()
                    print(f"Training output: {output[:1000]}...")  # First 1000 chars
            except:
                print("Could not read training output")
        else:
            print("Process is still running successfully")

except Exception as e:
    print(f"Error starting training: {str(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
    raise
"""
            
            response = self.sandbox.process.code_run(start_command)
            self.log_sandbox_response(response, "TRAINING_START")
            
            if response.exit_code != 0:
                self.log_operation("TRAINING_START_FAILED", {
                    "exit_code": response.exit_code,
                    "output": response.result
                })
                return False
            
            self.logger.info("Training started successfully in background")
            self.logger.info(f"Start response: {response.result}")
            
            # Monitor training progress
            return self.monitor_training()
                
        except Exception as e:
            self.log_operation("TRAINING_ERROR", error=e)
            return False

    def monitor_training(self):
        """Monitor the background training process"""
        import time
        
        try:
            self.log_operation("TRAINING_MONITORING_START")
            max_wait_time = 3600  # 1 hour max
            check_interval = 30   # Check every 30 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                self.logger.info(f"Checking training status... (elapsed: {elapsed_time}s)")
                
                # Check if process is still running
                status_command = """
import os
import subprocess

try:
    # Check if PID file exists
    if os.path.exists('training_pid.txt'):
        with open('training_pid.txt', 'r') as f:
            pid = f.read().strip()
        
        # Check if process is still running
        try:
            os.kill(int(pid), 0)  # Signal 0 checks if process exists
            print(f"RUNNING: Process {pid} is still active")
            
            # Show last few lines of training log
            if os.path.exists('training_output.log'):
                with open('training_output.log', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("=== RECENT TRAINING OUTPUT ===")
                        for line in lines[-5:]:  # Last 5 lines
                            print(line.strip())
                        print("=== END RECENT OUTPUT ===")
            
        except OSError:
            print(f"COMPLETED: Process {pid} has finished")
            
            # Check final output
            if os.path.exists('training_output.log'):
                with open('training_output.log', 'r') as f:
                    content = f.read()
                    print("=== FINAL TRAINING OUTPUT ===")
                    print(content[-2000:])  # Last 2000 characters
                    print("=== END FINAL OUTPUT ===")
            
            exit(0)  # Signal completion
            
    else:
        print("ERROR: PID file not found")
        exit(1)
        
except Exception as e:
    print(f"ERROR monitoring training: {e}")
    exit(1)
"""
                
                response = self.sandbox.process.code_run(status_command)
                self.log_sandbox_response(response, f"TRAINING_STATUS_CHECK_{elapsed_time}")
                
                if response.exit_code == 0:  # Training completed
                    self.log_operation("TRAINING_COMPLETED", {
                        "total_time": elapsed_time,
                        "final_output": response.result
                    })
                    return True
                elif "RUNNING" in response.result:
                    self.logger.info(f"Training still in progress... waiting {check_interval}s")
                    self.logger.debug(f"Status: {response.result}")
                else:
                    self.log_operation("TRAINING_MONITORING_ERROR", {
                        "elapsed_time": elapsed_time,
                        "status_output": response.result
                    })
                    
                time.sleep(check_interval)
                elapsed_time += check_interval
            
            # Timeout reached
            self.log_operation("TRAINING_TIMEOUT", {
                "max_wait_time": max_wait_time,
                "elapsed_time": elapsed_time
            })
            self.logger.warning(f"Training monitoring timed out after {max_wait_time}s")
            return False
            
        except Exception as e:
            self.log_operation("TRAINING_MONITORING_ERROR", error=e)
            return False

    def get_training_logs(self):
        """Retrieve training logs from the sandbox"""
        try:
            self.log_operation("RETRIEVING_TRAINING_LOGS")
            
            logs_command = """
import os

print("=== TRAINING LOG RETRIEVAL ===")

# Check if training output log exists
if os.path.exists('training_output.log'):
    print("Found training_output.log")
    with open('training_output.log', 'r') as f:
        content = f.read()
        print(f"Log size: {len(content)} characters")
        print("=== TRAINING OUTPUT START ===")
        print(content)
        print("=== TRAINING OUTPUT END ===")
else:
    print("training_output.log not found")

# Check if PID file exists
if os.path.exists('training_pid.txt'):
    with open('training_pid.txt', 'r') as f:
        pid = f.read().strip()
    print(f"Training PID was: {pid}")
else:
    print("training_pid.txt not found")

print("=== END LOG RETRIEVAL ===")
"""
            
            response = self.sandbox.process.code_run(logs_command)
            self.log_sandbox_response(response, "TRAINING_LOGS")
            
            if response.exit_code == 0:
                self.log_operation("TRAINING_LOGS_RETRIEVED", {
                    "logs": response.result
                })
                return response.result
            else:
                self.log_operation("TRAINING_LOGS_ERROR", {
                    "exit_code": response.exit_code,
                    "output": response.result
                })
                return None
                
        except Exception as e:
            self.log_operation("TRAINING_LOGS_EXCEPTION", error=e)
            return None

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
    
    # Initialize logging for main execution
    logger = logging.getLogger(__name__)
    error_logger = logging.getLogger('errors')
    
    try:
        logger.info("🚀 STARTING DAYTONA LLM TRAINING DEPLOYMENT")
        
        # Step 1: Create sandbox
        logger.info("📦 Step 1: Creating sandbox...")
        trainer.create_sandbox()
        
        # Step 2: Upload only essential files (fast)
        logger.info("📁 Step 2: Uploading essential files...")
        trainer.upload_essential_files()
        
        # Step 3: Install/verify dependencies (fail-fast approach)
        logger.info("🔧 Step 3: Verifying dependencies...")
        if not trainer.install_dependencies():
            logger.error("❌ Dependency verification failed. Stopping deployment.")
            raise Exception("Dependency verification failed")
        
        # Step 4: Upload all books (slower but needed for training)
        logger.info("📚 Step 4: Uploading training books...")
        trainer.upload_books()
        
        # Step 5: Start training (background mode with monitoring)
        logger.info("🧠 Step 5: Starting model training in background...")
        training_success = trainer.run_training()
        
        # Always retrieve training logs for analysis (even if training failed/timed out)
        logger.info("📋 Retrieving training logs...")
        training_logs = trainer.get_training_logs()
        if training_logs:
            logger.info("Training logs retrieved successfully")
        
        if not training_success:
            logger.warning("⚠️ Training monitoring failed or timed out")
            logger.info("💡 Training may still be running. Check logs above for status.")
            # Don't raise exception - continue to try downloading model
        else:
            logger.info("✅ Training completed successfully!")
        
        # Step 6: Download trained model (attempt even if training monitoring failed)
        logger.info("⬇️ Step 6: Attempting to download trained model...")
        try:
            trainer.download_model()
            logger.info("📦 Model download completed")
        except Exception as download_error:
            logger.warning(f"⚠️ Model download failed: {download_error}")
            logger.info("💡 Model may not be ready yet if training is still in progress")
        
        logger.info("🎉 DEPLOYMENT PROCESS COMPLETED!")
        logger.info("📋 Check training logs above for detailed training status")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Deployment interrupted by user (Ctrl+C)")
        error_logger.error("Deployment interrupted by user")
        
        # Try to get training logs before cleanup
        try:
            logger.info("📋 Attempting to retrieve training logs before cleanup...")
            training_logs = trainer.get_training_logs()
        except:
            logger.warning("Could not retrieve training logs during interrupt cleanup")
        
    except Exception as e:
        logger.error(f"💥 DEPLOYMENT FAILED: {str(e)}")
        error_logger.error(f"DEPLOYMENT_FAILED: {str(e)}")
        
        # Try to get training logs even on failure
        try:
            logger.info("📋 Attempting to retrieve training logs before cleanup...")
            training_logs = trainer.get_training_logs()
        except:
            logger.warning("Could not retrieve training logs during error cleanup")
        
    finally:
        # Always cleanup
        logger.info("🧹 Cleaning up resources...")
        trainer.cleanup()
        
        logger.info("📋 Check the following log files for detailed information:")
        logger.info("   - daytona_llm_trainer.log (full execution log)")
        logger.info("   - daytona_errors.log (errors only)")