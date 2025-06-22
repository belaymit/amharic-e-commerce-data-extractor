#!/usr/bin/env python3
"""
Main script to execute Tasks 3, 4, and 5 for Amharic NER
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split

# Transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')


def load_conll_data(file_path):
    """Load CoNLL format data"""
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('#'):
                continue
                
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_sentence.append(token)
                    current_labels.append(label)
    
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    
    return sentences, labels


def main():
    """Main execution function"""
    print("="*60)
    print("AMHARIC E-COMMERCE NER PIPELINE")
    print("Tasks 3, 4, and 5 Implementation")
    print("="*60)
    
    # Load data
    data_path = "data/conll_labeled/amharic_ecommerce_conll.txt"
    sentences, labels = load_conll_data(data_path)
    
    print(f"Loaded {len(sentences)} sentences")
    print("Data ready for model training!")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("\nTo run individual tasks:")
    print("1. Task 3: Open notebooks/NER_Fine_Tuning.ipynb")
    print("2. Task 4: Open notebooks/Model_Comparison.ipynb") 
    print("3. Task 5: Open notebooks/Model_Interpretability.ipynb")
    
    print("\nAll notebooks are ready for execution!")


if __name__ == "__main__":
    main() 