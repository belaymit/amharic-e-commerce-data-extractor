# Task 4: Model Comparison

## 🎯 Objective
Compare multiple NER models to select the best performing one for Amharic e-commerce entity extraction.

## 🤖 Models to Compare
1. **XLM-Roberta** (`xlm-roberta-base`)
   - Multilingual transformer
   - Strong cross-lingual performance
   - Best for accuracy

2. **DistilBERT** (`distilbert-base-multilingual-cased`)
   - Lightweight, faster training
   - Good balance of speed/performance
   - Best for production deployment

3. **mBERT** (`bert-base-multilingual-cased`)
   - Established multilingual model
   - Wide language support
   - Baseline comparison

## 📝 Implementation

### Notebook Location
```bash
notebooks/Model_Comparison.ipynb
```

### Automated Comparison
The notebook provides automated training and evaluation of all three models.

## 📊 Comparison Metrics

### Primary Metrics
- **F1-Score**: Harmonic mean of precision/recall
- **Precision**: Correct entities / Predicted entities
- **Recall**: Correct entities / True entities

### Secondary Metrics
- **Training Time**: Time to complete training
- **Model Size**: Memory footprint
- **Inference Speed**: Predictions per second

## 🔧 Evaluation Framework

### Cross-Validation
- **Data Split**: 80% train, 20% validation
- **Stratified**: Balanced entity distribution
- **Reproducible**: Fixed random seed

### Per-Entity Analysis
- Performance breakdown by entity type:
  - PRODUCT entities
  - PRICE entities
  - LOC entities

## 📈 Expected Performance

### Target Benchmarks
| Model | F1-Score | Training Time | Model Size |
|-------|----------|---------------|------------|
| XLM-Roberta | >0.85 | ~15 min | ~560MB |
| DistilBERT | >0.80 | ~8 min | ~270MB |
| mBERT | >0.82 | ~12 min | ~420MB |

### Selection Criteria
1. **Accuracy First**: F1-score > 0.80
2. **Speed Consideration**: Training time < 20 min
3. **Resource Efficiency**: Model size manageable

## 🚀 Usage

### Interactive Analysis
```bash
cd notebooks/
jupyter notebook Model_Comparison.ipynb
```

### Automated Execution
Run all cells in sequence for complete comparison

## 📂 Output Files

### Results
- **Comparison Report**: `models/comparison_results.csv`
- **Performance Plots**: `models/performance_plots.png`
- **Best Model**: `models/best-amharic-ner/`

### Model Artifacts
Each model produces:
- Trained weights
- Tokenizer files
- Label mappings
- Training logs

## 🎯 Model Selection Process

### 1. Quantitative Analysis
- F1-score comparison
- Training efficiency
- Resource requirements

### 2. Qualitative Analysis
- Error analysis on validation set
- Entity-specific performance
- Edge case handling

### 3. Final Selection
- Best overall performer
- Production readiness
- Deployment considerations

## 📊 Visualization

### Performance Charts
- F1-score comparison bar chart
- Training time comparison
- Per-entity performance heatmap

### Learning Curves
- Training/validation loss
- F1-score progression
- Convergence analysis

## 🔄 Integration

### With Task 3
- Uses fine-tuned models from Task 3
- Leverages same data preprocessing
- Consistent evaluation metrics

### With Task 5
- Best model feeds into interpretability
- Performance insights inform analysis
- Model selection validated

## 🎯 Success Criteria
- [ ] All three models trained successfully
- [ ] Performance comparison completed
- [ ] Best model identified and saved
- [ ] Results documented and visualized

## 📈 Optimization Tips

### For Better Performance
1. **Hyperparameter Tuning**: Grid search on learning rate
2. **Data Augmentation**: Synthetic sentence generation
3. **Ensemble Methods**: Combine multiple models

### For Faster Training
1. **Gradient Accumulation**: Larger effective batch sizes
2. **Mixed Precision**: FP16 training
3. **Model Parallelization**: Multi-GPU training

## 🐛 Troubleshooting
- **Memory Issues**: Reduce batch size or sequence length
- **Slow Training**: Use gradient checkpointing
- **Poor Performance**: Check data quality and labeling

## 🔄 Next Steps
1. Complete model comparison analysis
2. Select best performing model
3. Proceed to Task 5 (Model Interpretability)
4. Document findings and recommendations 