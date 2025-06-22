#!/usr/bin/env python3
"""
Demo script for Task 3: NER Fine-tuning
This script demonstrates the complete Task 3 workflow with a minimal training setup
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
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
    print("="*60)
    print("TASK 3 DEMO: AMHARIC NER FINE-TUNING")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = "data/conll_labeled/amharic_ecommerce_conll.txt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure Task 2 is completed.")
        return
    
    sentences, labels = load_conll_data(data_path)
    print(f"Loaded {len(sentences)} sentences")
    
    # Create label mappings
    unique_labels = set()
    for label_list in labels:
        unique_labels.update(label_list)
    
    label_list = sorted(list(unique_labels))
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    print(f"Labels: {label_list}")
    
    # Convert labels to numeric IDs
    label_ids = [[label2id[label] for label in label_list] for label_list in labels]
    
    # Split data - use smaller subset for demo
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences[:20], label_ids[:20], test_size=0.3, random_state=42
    )
    
    print(f"Demo training set: {len(train_sentences)} sentences")
    print(f"Demo validation set: {len(val_sentences)} sentences")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'tokens': train_sentences,
        'ner_tags': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'tokens': val_sentences,
        'ner_tags': val_labels
    })
    
    # Model setup - using DistilBERT for faster demo
    model_name = "distilbert-base-multilingual-cased"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenization function
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=256  # Shorter for demo
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )
    
    # Evaluation metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions),
            'f1': f1_score(true_labels, true_predictions),
        }
        return results
    
    # Training arguments - minimal for demo
    training_args = TrainingArguments(
        output_dir="models/demo-amharic-ner",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,  # Just 1 epoch for demo
        weight_decay=0.01,
        eval_strategy="epoch",  # Updated parameter name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs/demo",
        logging_steps=5,
        save_total_limit=1,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    start_time = time.time()
    print("\nStarting training...")
    trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall: {eval_results['eval_recall']:.4f}")
    
    # Save model
    model_path = "models/demo-amharic-ner-final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save label mappings
    with open(f"{model_path}/label_mappings.json", 'w', encoding='utf-8') as f:
        json.dump({'id2label': id2label, 'label2id': label2id}, f, ensure_ascii=False, indent=2)
    
    print(f"Model saved to: {model_path}")
    
    # Test on sample text
    def predict_entities(text):
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**tokens)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class = torch.argmax(predictions, dim=-1)
        
        predicted_labels = [id2label[pred.item()] for pred in predicted_token_class[0]]
        tokens_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
        
        # Filter special tokens
        filtered_tokens_labels = []
        for token, label in zip(tokens_list, predicted_labels):
            if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                filtered_tokens_labels.append((token, label))
        
        return filtered_tokens_labels
    
    # Test the model
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL")
    print("="*60)
    
    test_text = "ቦርሳ በጣም ጥሩ! ዋጋ 5000 ብር። ቦሌ ውስጥ ይገኛል።"
    print(f"Test text: {test_text}")
    
    predictions = predict_entities(test_text)
    print("\nPredicted entities:")
    for token, label in predictions:
        if label != 'O':
            print(f"  {token:<15} -> {label}")
    
    print("\n" + "="*60)
    print("TASK 3 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Next steps:")
    print("1. Run full training with more epochs")
    print("2. Compare with other models (Task 4)")
    print("3. Analyze model interpretability (Task 5)")
    
    return eval_results

if __name__ == "__main__":
    main()