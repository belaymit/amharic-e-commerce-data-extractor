# Task 5: Model Interpretability

## 🎯 Objective
Analyze and interpret the best-performing NER model using SHAP and LIME to understand decision-making patterns and build trust in the model.

## 🔍 Interpretability Goals
1. **Global Understanding**: Overall model behavior
2. **Local Explanations**: Individual prediction reasoning
3. **Feature Importance**: Critical input patterns
4. **Error Analysis**: Common failure modes
5. **Bias Detection**: Fairness and equity assessment

## 🛠️ Tools & Techniques

### 1. SHAP (SHapley Additive exPlanations)
- **Purpose**: Global and local interpretability
- **Method**: Game-theoretic approach
- **Output**: Feature importance scores
- **Best for**: Token-level analysis

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- **Purpose**: Local explanations
- **Method**: Surrogate model approach
- **Output**: Feature relevance
- **Best for**: Sentence-level analysis

## 📝 Implementation

### Notebook Location
```bash
notebooks/Model_Interpretability.ipynb
```

### Analysis Components
- SHAP value computation
- LIME explanations
- Attention visualization
- Error pattern analysis

## 🔧 Analysis Framework

### 1. Model Loading
```python
# Load the best model from Task 4
model = AutoModelForTokenClassification.from_pretrained(best_model_path)
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
```

### 2. SHAP Analysis
- **Token-level importance**: Which tokens drive predictions
- **Entity-level analysis**: How entities are recognized
- **Global patterns**: Overall model behavior

### 3. LIME Analysis
- **Local explanations**: Why specific predictions were made
- **Feature perturbation**: Impact of token changes
- **Prediction confidence**: Certainty levels

## 📊 Interpretability Outputs

### SHAP Visualizations
1. **Summary Plot**: Feature importance overview
2. **Waterfall Plot**: Individual prediction breakdown
3. **Force Plot**: Prediction contributions
4. **Partial Dependence**: Feature effect patterns

### LIME Explanations
1. **Local Explanations**: Per-prediction reasoning
2. **Feature Weights**: Token contribution scores
3. **Confidence Intervals**: Prediction uncertainty
4. **Alternative Predictions**: What-if scenarios

## 🎯 Analysis Focus Areas

### Entity Recognition Patterns
- **PRODUCT**: What makes text recognizable as products
- **PRICE**: Price pattern recognition logic
- **LOC**: Location identification strategies

### Language-Specific Insights
- **Amharic Script**: Unique character importance
- **Word Boundaries**: Subword tokenization effects
- **Context Dependencies**: Multi-token entity handling

### Model Behavior
- **Attention Patterns**: What the model focuses on
- **Decision Boundaries**: Entity vs non-entity distinctions
- **Confidence Calibration**: Prediction reliability

## 🚀 Usage

### Interactive Analysis
```bash
cd notebooks/
jupyter notebook Model_Interpretability.ipynb
```

### Example Analysis
```python
# Analyze specific sentence
text = "ቦርሳ በጣም ጥሩ! ዋጋ 5000 ብር። ቦሌ ውስጥ ይገኛል።"
explain_prediction(model, tokenizer, text)
```

## 📂 Output Files

### SHAP Results
- **Global Plots**: `interpretability/shap_summary.png`
- **Feature Importance**: `interpretability/feature_importance.json`
- **Entity Analysis**: `interpretability/entity_patterns.html`

### LIME Results
- **Local Explanations**: `interpretability/lime_explanations.html`
- **Confidence Analysis**: `interpretability/confidence_scores.csv`
- **Error Cases**: `interpretability/error_analysis.json`

### Reports
- **Interpretability Report**: `interpretability/analysis_report.md`
- **Model Insights**: `interpretability/model_insights.json`
- **Recommendations**: `interpretability/recommendations.md`

## 🔍 Key Insights Expected

### Model Strengths
- Effective pattern recognition
- Robust entity boundary detection
- Good context understanding

### Model Limitations
- Potential biases
- Failure modes
- Confidence calibration issues

### Improvement Opportunities
- Data augmentation needs
- Model architecture adjustments
- Training strategy refinements

## 📈 Interpretability Metrics

### Explanation Quality
- **Fidelity**: How well explanations match model behavior
- **Stability**: Consistency across similar examples
- **Comprehensibility**: Human understandability

### Model Trust
- **Transparency**: Clear decision reasoning
- **Reliability**: Consistent performance patterns
- **Fairness**: Unbiased treatment across entities

## 🔄 Integration

### With Task 3 & 4
- Uses best model from comparison
- Analyzes training effectiveness
- Validates model selection

### For Production
- Identifies deployment risks
- Guides monitoring strategies
- Informs user communication

## 🎯 Success Criteria
- [ ] SHAP analysis completed successfully
- [ ] LIME explanations generated
- [ ] Global and local patterns identified
- [ ] Model behavior documented
- [ ] Interpretability report created

## 📊 Analysis Examples

### PRODUCT Entity Recognition
```
Input: "ቦርሳ በጣም ጥሩ"
SHAP: ቦርስ(+0.8), ባ(+0.2), በ(-0.1), ጣም(-0.05), ጥሩ(+0.1)
LIME: "ቦርሳ" drives PRODUCT prediction (confidence: 0.92)
```

### PRICE Entity Recognition
```
Input: "ዋጋ 5000 ብር"
SHAP: ዋ(+0.3), ጋ(+0.2), 5000(+0.9), ብር(+0.7)
LIME: Numeric pattern + currency word = PRICE (confidence: 0.95)
```

## 🐛 Troubleshooting

### SHAP Issues
- **Memory**: Use smaller batch sizes
- **Speed**: Sample subset of data
- **Visualization**: Reduce plot complexity

### LIME Issues
- **Perturbation**: Adjust sampling strategy
- **Explanations**: Tune surrogate model
- **Stability**: Increase sampling size

## 📚 Theoretical Background

### SHAP Theory
- Shapley values from game theory
- Additive feature attribution
- Efficient approximation algorithms

### LIME Theory
- Local linear approximation
- Feature perturbation sampling
- Surrogate model training

## 🔄 Next Steps
1. Complete interpretability analysis
2. Document key findings and insights
3. Create deployment recommendations
4. Integrate insights into production pipeline
5. Plan ongoing monitoring and analysis

## 💡 Business Impact
- **Trust**: Stakeholder confidence in model
- **Debugging**: Identify and fix issues
- **Compliance**: Meet interpretability requirements
- **Performance**: Guide model improvements 