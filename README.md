# Electronics LLM Training Platform

![License](https://img.shields.io/badge/License-MIT-blue.svg)

A specialized platform for training language models on electronics textbooks using Daytona cloud infrastructure. Features intelligent memory management, complete book processing, and optimized training for technical domain expertise.

## Overview

This platform automates the entire pipeline from PDF processing to model deployment, including memory-aware text extraction, intelligent resource management, and fine-tuning of transformer models for electronics domain applications.

## Key Features

### Intelligent Document Processing
- **Memory-Aware Extraction**: Processes 70-90% of each book within memory limits
- **Complete Book Coverage**: No arbitrary page limits - extracts maximum content safely
- **Smart Resource Management**: 75% memory threshold prevents crashes
- **Real-time Monitoring**: Detailed memory usage and coverage reporting

### Optimized Model Training
- **DialoGPT-Medium (345M params)**: High-quality electronics expert model
- **Quality-First Settings**: Longer sequences, conservative learning rates
- **CPU-Optimized**: Efficient training without GPU requirements
- **Progress Tracking**: Detailed logging of training metrics and checkpoints

### Cloud Integration
- **Daytona Platform**: Seamless deployment to cloud sandboxes
- **Automatic Download**: Trained models delivered to local machine
- **Cost Efficiency**: Optimized resource usage within credit budget
- **Error Recovery**: Robust error handling and automatic retries

## File Structure

- **`deploy.py`** - Main deployment script (starts training)
- **`trainer.py`** - Core training logic with memory management
- **`trainer_old.py`** - Legacy complex trainer (backup)
- **`requirements.txt`** - Package dependencies
- **`books/`** - Place your electronics PDF books here
- **`trained_model/`** - Downloaded trained model files

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Daytona account with API access
- PDF documents for training (place in `books/` directory)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd daytona-llm-trainer
   ```

2. **Configure Environment**
   Create a `.env` file with your configuration:
   ```bash
   # Add your actual Daytona API key here
   daytona_key=your_daytona_api_key_here
   ```

3. **Prepare Training Data**
   Place your PDF documents in the `books/` directory. The system will automatically process all PDF files found in this location.

### Usage

#### Basic Training
```bash
python deploy.py
```

This command will:
- Create a cloud sandbox with required dependencies
- Upload your documents and training scripts
- Process PDFs with memory-aware text extraction
- Train the DialoGPT-medium model on electronics content
- Download the trained model upon completion

#### Local Testing (Development)
```bash
python trainer.py
```

For local development and testing without cloud deployment.

## System Architecture

### Processing Pipeline
1. **Document Upload**: PDFs are uploaded to the cloud sandbox
2. **Content Extraction**: Text, images, and metadata are extracted
3. **OCR Processing**: Image-embedded text is recognized and extracted
4. **Content Consolidation**: All extracted content is merged and validated
5. **Dataset Preparation**: Content is tokenized and formatted for training
6. **Model Training**: Fine-tuning is performed on the prepared dataset
7. **Model Export**: Trained models are saved and made available for download

### Resource Configuration
- **CPU**: 4 cores for parallel processing
- **Memory**: 4GB RAM optimized for efficiency
- **Storage**: 10GB with automatic cleanup
- **Training Time**: Typically 30-60 minutes depending on content volume

## Configuration Options

### Training Parameters
Edit `electronics_llm_trainer.py` to customize:
- **Model Architecture**: Choose base model (default: DialoGPT-medium)
- **Training Duration**: Number of epochs and batch size
- **Learning Rate**: Optimization parameters
- **Content Filtering**: Criteria for document relevance

### Processing Settings
Modify extraction settings:
- **OCR Quality**: DPI and preprocessing options
- **Image Analysis**: Computer vision model selection
- **Text Processing**: Cleaning and normalization parameters
- **Memory Management**: Batch processing and cleanup intervals

## Monitoring and Logging

### Comprehensive Logging
- **Training Progress**: Real-time updates on model training
- **Processing Metrics**: Document processing statistics
- **Error Tracking**: Detailed error logs with full stack traces
- **Resource Usage**: CPU, memory, and storage utilization

### Log Files
- `daytona_llm_trainer.log`: Complete execution log
- `daytona_errors.log`: Error-specific logging
- `processing_summary.json`: Document processing statistics

## Advanced Features

### Multimodal Processing
The system combines multiple types of content extraction:
- **Plain Text**: Direct text extraction from PDFs
- **OCR Text**: Text recognition from images and diagrams
- **Image Analysis**: Understanding of visual content and diagrams
- **Metadata Extraction**: Document properties and structure

### Space Management
Automatic optimization to work within storage constraints:
- **Progressive Deletion**: Source documents removed after processing
- **Content Compression**: Extracted content stored efficiently
- **Usage Tracking**: Detailed metrics on space savings
- **Cleanup Automation**: Temporary files automatically managed

### Error Recovery
Robust handling of common issues:
- **Network Interruptions**: Automatic retry mechanisms
- **Memory Constraints**: Adaptive batch sizing
- **Processing Failures**: Skip problematic documents and continue
- **Timeout Management**: Background processing for long-running tasks

## Best Practices

### Document Preparation
- Ensure PDFs are not password-protected
- Use high-quality scans for better OCR results
- Organize documents by topic or domain for better training
- Remove duplicate or irrelevant content

### Resource Management
- Monitor training logs for progress and issues
- Adjust batch sizes based on available memory
- Use background training for large document sets
- Regular cleanup of temporary files

### Quality Assurance
- Review processing summaries for extraction quality
- Validate model outputs with test prompts
- Monitor training metrics for convergence
- Keep backups of successfully trained models

## Troubleshooting

### Common Issues
- **Memory Errors**: Reduce batch size or document count
- **OCR Failures**: Check image quality and format compatibility
- **Training Stalls**: Monitor for convergence issues or data problems
- **Upload Failures**: Verify API credentials and network connectivity

### Debug Mode
Enable detailed logging by setting debug flags in the configuration files. This provides additional information about each processing step.

## Contributing

This platform is designed for extensibility. Common modifications include:
- Adding new document formats
- Implementing different model architectures
- Enhancing content extraction algorithms
- Improving resource optimization strategies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 SysQuirrel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

**Ready to begin training?** Place your PDF documents in the `books/` directory and run `python deploy.py` to start the automated training pipeline.