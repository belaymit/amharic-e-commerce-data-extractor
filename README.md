# Amharic E-commerce Data Extractor

A comprehensive Python project for extracting and analyzing e-commerce data from Amharic Telegram channels. This project implements Named Entity Recognition (NER) capabilities to identify products, prices, and locations from Amharic text.

## 🎯 Features

- **Data Ingestion**: Scrape data from multiple Ethiopian Telegram e-commerce channels
- **Amharic NLP**: Process and analyze Amharic text with specialized preprocessing
- **NER Models**: Fine-tuned transformer models for entity extraction (XLM-Roberta, DistilBERT, mBERT)
- **Model Interpretability**: SHAP and LIME analysis for model transparency
- **FinTech Integration**: Vendor scoring for micro-lending applications

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd amharic-ecommerce-data-extractor
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📊 Entity Types Supported

- **Products**: ቦርሳ (bag), ጫማ (shoes), cream, etc.
- **Prices**: ዋጋ 5000 ብር, በ 1200 ብር patterns
- **Locations**: ቦሌ, አዲስ አበባ, መርካቶ, ፒያሳ

## 🚀 Quick Start

### Data Collection (Task 1)
```bash
python scripts/run_task1.py
```

### NER Model Training (Task 3)
```bash
python scripts/demo_task3_training.py
# Or run: jupyter notebook notebooks/NER_Fine_Tuning.ipynb
```

### Model Comparison (Task 4)
```bash
# Run in Jupyter notebooks/Model_Comparison.ipynb
```

### Model Interpretability (Task 5)
```bash
# Run in Jupyter notebooks/Model_Interpretability.ipynb
```

## 📁 Project Structure

```
├── src/
│   ├── core/          # Core pipeline and storage components
│   ├── models/        # ML model definitions
│   ├── utils/         # Preprocessing and utility functions
│   └── services/      # Telegram scrapers and external services
├── notebooks/         # Jupyter notebooks for tasks and analysis
├── scripts/           # Executable scripts for tasks
├── data/              # Data storage (raw and processed)
├── config/            # Configuration files
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── examples/          # Usage examples
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure your settings:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## 📚 Documentation

- See `docs/` for detailed documentation
- Check `notebooks/` for interactive examples
- Review `docs/Task3.md`, `docs/Task4.md`, `docs/Task5.md` for implementation details

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `MIT.md` file for details.

## 🙏 Acknowledgments

- Ethiopian Telegram e-commerce community
- Hugging Face for transformer models
- Contributors and maintainers
