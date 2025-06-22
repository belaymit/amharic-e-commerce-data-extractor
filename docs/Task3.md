# Task 3: NER Fine-Tuning

## 🎯 Objective
Fine-tune transformer models for Amharic Named Entity Recognition to extract products, prices, and locations from e-commerce messages.

## 📊 Dataset
- **Source**: CoNLL labeled Amharic e-commerce data
- **Size**: 50 sentences with BIO tagging
- **Entities**: 
  - B-PRODUCT, I-PRODUCT (products)
  - B-PRICE, I-PRICE (prices)
  - B-LOC, I-LOC (locations)
  - O (outside entities)

## 🤖 Models
1. **XLM-Roberta** - Multilingual transformer (primary)
2. **DistilBERT** - Faster, lightweight option
3. **mBERT** - Multilingual BERT

## 📝 Implementation

### Notebook Location
```bash
notebooks/NER_Fine_Tuning.ipynb
```

### Quick Demo
```bash
python scripts/demo_task3_training.py
```

## 🔧 Key Components

### 1. Data Loading
```python
def load_conll_data(file_path):
    """Load CoNLL format data with BIO tagging"""
    # Implementation in notebook
```

### 2. Model Setup
- Tokenizer alignment for subword handling
- Label mapping (id2label, label2id)
- Token classification head

### 3. Training Configuration
- Learning rate: 3e-5
- Batch size: 8 (16 for full training)
- Epochs: 3+ (1 for demo)
- Evaluation strategy: per epoch

### 4. Metrics
- **Precision**: Entity-level precision
- **Recall**: Entity-level recall  
- **F1-Score**: Harmonic mean (target: >0.85)

## 📈 Expected Results

### Demo Results (Minimal Training)
- Training time: ~7 seconds
- F1-Score: 0.0 (expected with 1 epoch)
- Purpose: Validate pipeline

### Full Training Results
- Training time: 5-15 minutes
- F1-Score: >0.85 (target)
- Precision: >0.80
- Recall: >0.80

## 🚀 Usage

### 1. Interactive Notebook
```bash
cd notebooks/
jupyter notebook NER_Fine_Tuning.ipynb
```

### 2. Command Line Demo
```bash
python scripts/demo_task3_training.py
```

### 3. Full Training
Run all cells in the notebook with increased epochs

## 📂 Output Files
- **Model**: `models/amharic-ner-model/`
- **Tokenizer**: `models/amharic-ner-model/`
- **Mappings**: `models/amharic-ner-model/label_mappings.json`
- **Logs**: `logs/training/`

## 🎯 Entity Examples
- **Products**: ቦርሳ, ጫማ, cream, iPhone
- **Prices**: ዋጋ 5000 ብር, በ 1200 ብር
- **Locations**: ቦሌ, አዲስ አበባ, መርካቶ

## 🔄 Next Steps
1. Complete Task 3 training
2. Proceed to Task 4 (Model Comparison)
3. Analyze with Task 5 (Model Interpretability)

## 🐛 Troubleshooting
- **CUDA**: Use CPU if GPU unavailable
- **Memory**: Reduce batch size if OOM
- **Performance**: Increase epochs for better results 