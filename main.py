#!/usr/bin/env python3
"""
Main entry point for Amharic E-commerce Data Extractor
Combines all tasks (1-6) into a single executable script
"""

import os
import sys
import json
import time
import asyncio
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core imports
try:
    from src.pipeline import run_task_1
    from src.config import app_config
except ImportError:
    print("⚠️  Warning: Core pipeline modules not found. Task 1 may not work.")
    run_task_1 = None
    app_config = None

# ML imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        TrainingArguments, 
        Trainer,
        DataCollatorForTokenClassification
    )
    from datasets import Dataset
    from seqeval.metrics import f1_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class AmharicNERPipeline:
    """Main pipeline for Amharic E-commerce Data Extraction"""
    
    def __init__(self):
        self.data_path = "data/conll_labeled/amharic_ecommerce_conll.txt"
        self.models_dir = "models"
        self.logs_dir = "logs"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/vendor_analytics", exist_ok=True)
    
    def load_conll_data(self, file_path):
        """Load CoNLL format data"""
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found.")
            print("Please ensure Task 2 (CoNLL labeling) is completed first.")
            return None, None
        
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
    
    async def run_task_1(self):
        """Task 1: Data Collection and Preprocessing"""
        print("=" * 60)
        print("🇪🇹 TASK 1: DATA COLLECTION & PREPROCESSING")
        print("=" * 60)
        
        if not run_task_1:
            print("❌ Task 1 pipeline not available. Please check src/pipeline.py")
            return False
        
        print("Starting data ingestion from Telegram channels...")
        
        try:
            # Configuration
            channels = app_config.target_channels if app_config else [
                "@ethio_market", "@addis_shopping", "@ethio_commerce"
            ]
            limit_per_channel = 500
            days_back = 7
            
            print(f"📡 Target channels: {', '.join(channels)}")
            print(f"📊 Limit per channel: {limit_per_channel}")
            print(f"📅 Days back: {days_back}")
            print()
            
            # Run the pipeline
            results = await run_task_1(
                channels=channels,
                limit_per_channel=limit_per_channel,
                days_back=days_back,
                export_csv=True
            )
            
            # Display results
            print("✅ Task 1 completed successfully!")
            print(f"📈 Total messages: {results.get('total_messages_processed', 0)}")
            print(f"📊 Channels processed: {results.get('channels_processed', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Task 1 failed: {e}")
            return False
    
    def run_task_2_check(self):
        """Check if Task 2 (CoNLL labeling) is completed"""
        print("=" * 60)
        print("📝 TASK 2: DATA LABELING CHECK")
        print("=" * 60)
        
        if os.path.exists(self.data_path):
            sentences, labels = self.load_conll_data(self.data_path)
            if sentences:
                print(f"✅ Task 2 completed: {len(sentences)} labeled sentences found")
                
                # Show label statistics
                unique_labels = set()
                for label_list in labels:
                    unique_labels.update(label_list)
                
                print(f"📊 Entity types: {sorted(list(unique_labels))}")
                return True
        
        print("❌ Task 2 not completed. Please run CoNLL labeling first.")
        print("💡 Use: jupyter notebook notebooks/CoNLL_Labeling.ipynb")
        return False
    
    def run_task_3(self, quick_demo=True):
        """Task 3: NER Model Fine-tuning"""
        print("=" * 60)
        print("🤖 TASK 3: NER MODEL FINE-TUNING")
        print("=" * 60)
        
        if not ML_AVAILABLE:
            print("❌ ML libraries not available. Please install requirements.txt")
            return False
        
        # Load data
        sentences, labels = self.load_conll_data(self.data_path)
        if not sentences:
            return False
        
        print(f"📊 Loaded {len(sentences)} sentences")
        
        # Create label mappings
        unique_labels = set()
        for label_list in labels:
            unique_labels.update(label_list)
        
        label_list = sorted(list(unique_labels))
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}
        
        print(f"🏷️  Labels: {label_list}")
        
        # Convert labels to numeric IDs
        label_ids = [[label2id[label] for label in label_list] for label_list in labels]
        
        # Data split
        if quick_demo:
            # Use smaller subset for demo
            subset_size = min(20, len(sentences))
            sentences = sentences[:subset_size]
            label_ids = label_ids[:subset_size]
            print(f"🚀 Quick demo mode: using {subset_size} sentences")
        
        train_sentences, val_sentences, train_labels, val_labels = train_test_split(
            sentences, label_ids, test_size=0.3, random_state=42
        )
        
        print(f"📚 Training set: {len(train_sentences)} sentences")
        print(f"📖 Validation set: {len(val_sentences)} sentences")
        
        # Model setup
        model_name = "distilbert-base-multilingual-cased"  # Faster for demo
        print(f"🔧 Loading model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
        
        # Tokenization function
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding=False,
                max_length=256
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
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'tokens': train_sentences,
            'ner_tags': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'tokens': val_sentences,
            'ner_tags': val_labels
        })
        
        # Tokenize datasets
        print("🔄 Tokenizing datasets...")
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
        
        # Training arguments
        epochs = 1 if quick_demo else 3
        output_dir = f"{self.models_dir}/amharic-ner-{'demo' if quick_demo else 'final'}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{self.logs_dir}/task3",
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
        
        # Training
        print("🚀 Starting training...")
        start_time = time.time()
        
        try:
            trainer.train()
            training_time = time.time() - start_time
            
            # Final evaluation
            eval_results = trainer.evaluate()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            print("✅ Task 3 completed successfully!")
            print(f"⏱️  Training time: {training_time:.2f} seconds")
            print(f"📊 Final F1-score: {eval_results.get('eval_f1', 0):.4f}")
            print(f"💾 Model saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def run_task_4(self):
        """Task 4: Model Comparison"""
        print("=" * 60)
        print("⚖️  TASK 4: MODEL COMPARISON")
        print("=" * 60)
        
        print("📊 Available for comparison:")
        print("1. XLM-Roberta (best performance)")
        print("2. DistilBERT (fast training)")
        print("3. mBERT (multilingual)")
        print()
        print("💡 For detailed comparison, use:")
        print("   jupyter notebook notebooks/Model_Comparison.ipynb")
        
        return True
    
    def run_task_5(self):
        """Task 5: Model Interpretability"""
        print("=" * 60)
        print("🔍 TASK 5: MODEL INTERPRETABILITY")
        print("=" * 60)
        
        print("🔬 Available interpretability tools:")
        print("1. SHAP - Feature importance analysis")
        print("2. LIME - Local explanations")
        print("3. Attention visualization")
        print()
        print("💡 For detailed analysis, use:")
        print("   jupyter notebook notebooks/Model_Interpretability.ipynb")
        
        return True
    
    def create_sample_vendor_data(self):
        """Create sample vendor data if no database exists"""
        sample_data = []
        
        # Sample Ethiopian e-commerce channels with realistic data
        vendors = [
            {
                'channel': '@EthioFashion',
                'title': 'Ethiopian Fashion Store',
                'messages': [
                    {'text': 'የሴቶች ቦርሳ ዋጋ 2500 ብር ቦሌ ውስጥ', 'views': 1250, 'date': '2025-01-15'},
                    {'text': 'ጫማ በጣም ጥሩ ዋጋ 3000 ብር', 'views': 890, 'date': '2025-01-14'},
                    {'text': 'ሻምፖ cream ዋጋ 450 ብር', 'views': 670, 'date': '2025-01-13'},
                    {'text': 'iPhone case ዋጋ 800 ብር መርካቶ', 'views': 1100, 'date': '2025-01-12'},
                    {'text': 'ልብስ ሽያጭ ዋጋ 1200 ብር', 'views': 780, 'date': '2025-01-11'},
                ]
            },
            {
                'channel': '@AddisMarket',
                'title': 'Addis Ababa Market',
                'messages': [
                    {'text': 'ሞባይል ፎን ዋጋ 15000 ብር ፒያሳ', 'views': 2100, 'date': '2025-01-15'},
                    {'text': 'ቦርሳ ሽያጭ ዋጋ 1800 ብር', 'views': 950, 'date': '2025-01-14'},
                    {'text': 'የወንዶች ሸሚዝ ዋጋ 1500 ብር', 'views': 1200, 'date': '2025-01-13'},
                ]
            },
            {
                'channel': '@BoleShop',
                'title': 'Bole Shopping Center',
                'messages': [
                    {'text': 'laptop ዋጋ 45000 ብር ቦሌ', 'views': 3200, 'date': '2025-01-15'},
                    {'text': 'ቦርሳ collection ዋጋ 2800 ብር', 'views': 1400, 'date': '2025-01-14'},
                ]
            }
        ]
        
        # Flatten to message-level data
        for vendor in vendors:
            for msg in vendor['messages']:
                sample_data.append({
                    'channel': vendor['channel'],
                    'channel_title': vendor['title'],
                    'text': msg['text'],
                    'views': msg['views'],
                    'date': msg['date'],
                    'entities': self.extract_entities_simple(msg['text'])
                })
        
        df = pd.DataFrame(sample_data)
        return df
    
    def extract_entities_simple(self, text):
        """Simple entity extraction for demo purposes"""
        entities = {'products': [], 'prices': [], 'locations': []}
        
        # Simple patterns for demo
        import re
        
        # Price patterns - more comprehensive
        price_patterns = [
            r'ዋጋ\s*(\d+)\s*ብር',
            r'በ\s*(\d+)\s*ብር',
            r'(\d+)\s*ብር'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            entities['prices'].extend([match for match in matches])
        
        # Product patterns (common words)
        product_words = ['ቦርሳ', 'ጫማ', 'ልብስ', 'ሸሚዝ', 'ሻምፖ', 'ሞባይል', 'ፎን', 'iPhone', 'laptop', 'cream', 'case', 'collection']
        for word in product_words:
            if word in text:
                entities['products'].append(word)
        
        # Location patterns
        location_words = ['ቦሌ', 'መርካቶ', 'ፒያሳ', 'አዲስ', 'አበባ', 'ውስጥ']
        for word in location_words:
            if word in text:
                entities['locations'].append(word)
        
        return json.dumps(entities)
    
    def run_task6_vendor_scorecard(self):
        """
        Execute Task 6: FinTech Vendor Scorecard for Micro-Lending
        Uses modular analytics architecture for better maintainability
        """
        print("🏦 TASK 6: FinTech Vendor Scorecard for Micro-Lending")
        print("=" * 60)
        
        try:
            # Import modular analytics components
            from src.analytics import (
                VendorDataLoader,
                VendorAnalyticsEngine,
                RiskAssessment,
                ReportExporter
            )
            
            # Step 1: Load vendor data
            print("🔄 Loading vendor data...")
            data_loader = VendorDataLoader()
            df, data_source = data_loader.load_vendor_data()
            
            if not data_loader.validate_data_structure(df):
                print("❌ Data validation failed")
                return False
            
            data_summary = data_loader.get_data_summary(df)
            print(f"✅ Loaded {data_summary['total_messages']} messages from {data_summary['unique_vendors']} vendors")
            
            # Step 2: Calculate vendor analytics
            print("\n🧮 Calculating vendor analytics...")
            analytics_engine = VendorAnalyticsEngine()
            vendor_analytics = analytics_engine.process_all_vendors(df, verbose=False)
            metrics_df = analytics_engine.create_metrics_dataframe(vendor_analytics)
            
            # Step 3: Risk assessment and loan calculations
            print("🎯 Performing risk assessment...")
            risk_assessor = RiskAssessment()
            enhanced_metrics_df = risk_assessor.process_vendor_portfolio(metrics_df)
            portfolio_summary = risk_assessor.get_portfolio_risk_summary(enhanced_metrics_df)
            
            # Step 4: Generate reports
            print("💾 Generating comprehensive reports...")
            exporter = ReportExporter()
            exported_files = exporter.export_all_reports(enhanced_metrics_df, data_source, df)
            
            # Display results
            print(f"\n📊 PORTFOLIO ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"🏆 Top Performer: {enhanced_metrics_df.iloc[0]['channel_title']}")
            print(f"📈 Average Score: {enhanced_metrics_df['lending_score'].mean():.1f}/100")
            print(f"💰 Total Portfolio: {portfolio_summary['total_portfolio_value']:,} ETB")
            print(f"🎯 Approval Rate: {portfolio_summary['approval_rate']:.1f}%")
            print(f"📁 Reports Exported: {len([f for f in exported_files.values() if f])}")
            
            # Show sample scorecard
            print(f"\n🔍 SAMPLE VENDOR SCORECARD:")
            scorecard_sample = enhanced_metrics_df[['channel_title', 'lending_score', 'risk_category', 'recommended_loan_etb']].head(3)
            for idx, (_, vendor) in enumerate(scorecard_sample.iterrows(), 1):
                risk_emoji = vendor.get('risk_color', '⚪')
                print(f"   {idx}. {vendor['channel_title'][:25]:25} | Score: {vendor['lending_score']:5.1f} | {risk_emoji} {vendor['risk_category']:12} | Loan: {vendor['recommended_loan_etb']:,} ETB")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure all analytics modules are properly installed")
            return False
        except Exception as e:
            print(f"❌ Error in Task 6: {e}")
            return False
    
    def run_all_tasks(self):
        """Run all tasks in sequence"""
        print("🚀 RUNNING ALL TASKS")
        print("=" * 60)
        
        # Task 1
        try:
            asyncio.run(self.run_task_1())
            print("✅ Task 1: Data Collection - COMPLETED")
        except:
            print("⚠️  Task 1: Data Collection - SKIPPED/FAILED")
        
        print()
        
        # Task 2 Check
        if self.run_task_2_check():
            print("✅ Task 2: Data Labeling - COMPLETED")
        else:
            print("❌ Task 2: Data Labeling - REQUIRED")
            return False
        
        print()
        
        # Task 3
        if self.run_task_3(quick_demo=True):
            print("✅ Task 3: NER Training - COMPLETED")
        else:
            print("❌ Task 3: NER Training - FAILED")
            return False
        
        print()
        
        # Task 4
        self.run_task_4()
        print("✅ Task 4: Model Comparison - READY")
        
        print()
        
        # Task 5
        self.run_task_5()
        print("✅ Task 5: Model Interpretability - READY")
        
        print()
        
        # Task 6
        if self.run_task6_vendor_scorecard():
            print("✅ Task 6: Vendor Scorecard - COMPLETED")
        else:
            print("❌ Task 6: Vendor Scorecard - FAILED")
            return False
        
        print()
        print("🎉 ALL TASKS COMPLETED!")
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Amharic E-commerce Data Extractor - Complete Pipeline"
    )
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['1', '2', '3', '4', '5', '6', 'all'], 
        help='Specific task to run (1-6) or "all" for complete pipeline'
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick demo mode for faster execution'
    )
    parser.add_argument(
        '--interactive', 
        action='store_true', 
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AmharicNERPipeline()
    
    print("🇪🇹 AMHARIC E-COMMERCE DATA EXTRACTOR")
    print("=" * 60)
    print("A comprehensive NLP solution for Ethiopian e-commerce")
    print("=" * 60)
    print()
    
    # Interactive mode
    if args.interactive or not args.task:
        while True:
            print("\n📋 Available Tasks:")
            print("1. Data Collection (Telegram scraping)")
            print("2. Data Labeling Check (CoNLL format)")
            print("3. NER Model Training")
            print("4. Model Comparison")
            print("5. Model Interpretability")
            print("6. FinTech Vendor Scorecard")
            print("7. Run All Tasks")
            print("0. Exit")
            
            choice = input("\n🔢 Enter your choice (0-7): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                asyncio.run(pipeline.run_task_1())
            elif choice == '2':
                pipeline.run_task_2_check()
            elif choice == '3':
                pipeline.run_task_3(quick_demo=args.quick)
            elif choice == '4':
                pipeline.run_task_4()
            elif choice == '5':
                pipeline.run_task_5()
            elif choice == '6':
                pipeline.run_task6_vendor_scorecard()
            elif choice == '7':
                pipeline.run_all_tasks()
            else:
                print("❌ Invalid choice. Please try again.")
    
    # Direct task execution
    else:
        if args.task == '1':
            asyncio.run(pipeline.run_task_1())
        elif args.task == '2':
            pipeline.run_task_2_check()
        elif args.task == '3':
            pipeline.run_task_3(quick_demo=args.quick)
        elif args.task == '4':
            pipeline.run_task_4()
        elif args.task == '5':
            pipeline.run_task_5()
        elif args.task == '6':
            pipeline.run_task6_vendor_scorecard()
        elif args.task == 'all':
            pipeline.run_all_tasks()


if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    main() 