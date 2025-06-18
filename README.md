# Amharic E-commerce Data Extractor

A comprehensive system for collecting, processing, and analyzing Ethiopian e-commerce data from Telegram channels with **CoNLL format labeling for NER training**.

## 🎯 Project Overview

This project extracts entities (products, prices, locations) from Amharic text in Ethiopian e-commerce Telegram channels, processes them into training data, and creates CoNLL format labeled datasets for Named Entity Recognition (NER) models.

### Key Features
- **Historical Data Collection**: 2018-2025 comprehensive scraping
- **Multi-channel Support**: 4+ Ethiopian e-commerce channels  
- **Amharic Text Processing**: Entity extraction with regex patterns
- **CoNLL Format Labeling**: BIO tagging for NER training
- **Database Storage**: SQLite with optimized indexing
- **Data Visualization**: 6 different dashboard views
- **NER Training Ready**: Structured entity extraction and labeled data

## 📁 Project Structure

```
amharic-e-commerce-data-extractor/
├── notebook/                                    # Jupyter notebooks
│   ├── 01_setup_and_config.ipynb               # Environment setup (Task 1)
│   ├── 02_amharic_processing.ipynb             # Text processing (Task 1)
│   ├── 03_multi_channel_scraping.ipynb         # Recent data collection (Task 1)
│   ├── 04_data_processing.ipynb                # Message processing (Task 1)
│   ├── 05_export_and_analysis.ipynb            # Data export & analysis (Task 1)
│   ├── 06_data_visualization.ipynb             # Comprehensive visualization (Task 1)
│   ├── 07_historical_data_collection.ipynb     # 2018-2025 collection (Task 1)
│   ├── Task2_CoNLL_Labeling.ipynb              # CoNLL format labeling (Task 2)
│   └── NOTEBOOKS_README.md                     # Detailed notebook guide
├── src/                                        # Source code modules
│   ├── config/settings.py                     # Configuration management
│   ├── scrapers/telegram_scraper.py           # Telegram scraping
│   ├── preprocessing/amharic_processor.py     # Text processing
│   ├── pipeline/data_ingestion.py             # Data pipeline
│   └── storage/database.py                    # Database operations
├── data/                                       # Data storage
│   ├── raw/                                   # Raw scraped data
│   ├── processed/                             # Processed datasets
│   │   ├── amharic_ecommerce.db              # Main processed database
│   │   └── historical_messages_2018_2025.csv # Historical export
│   ├── conll_labeled/                         # Task 2 outputs
│   │   ├── amharic_ecommerce_conll.txt       # CoNLL format labeled data
│   │   ├── labeling_statistics.json          # Labeling statistics
│   │   └── validation_report.json            # Validation report
│   ├── demo.db                               # Demo database
│   ├── multi_channel_test.db                 # Multi-channel testing
│   └── historical_messages.db                # Historical data (2018-2025)
├── requirements.txt                           # Python dependencies
├── setup.py                                  # Package setup
├── run_task1.py                              # Task 1 execution script
└── .env.example                              # Environment variables template
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd amharic-e-commerce-data-extractor

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your Telegram API credentials
```

### 2. Task 1: Data Collection & Processing
```bash
# Historical data collection (2018-2025)
jupyter notebook notebook/07_historical_data_collection.ipynb

# Recent data pipeline (sequential execution)
jupyter notebook notebook/01_setup_and_config.ipynb
jupyter notebook notebook/02_amharic_processing.ipynb
jupyter notebook notebook/03_multi_channel_scraping.ipynb
jupyter notebook notebook/04_data_processing.ipynb
jupyter notebook notebook/05_export_and_analysis.ipynb

# Comprehensive visualization
jupyter notebook notebook/06_data_visualization.ipynb
```

### 3. Task 2: CoNLL Format Labeling
```bash
# CoNLL format labeling for NER training
jupyter notebook notebook/Task2_CoNLL_Labeling.ipynb
```

## 📊 Data Collection Capabilities (Task 1)

### Historical Coverage (2018-2025)
- **8 years** of comprehensive data
- **4 channels**: @ShegerOnlineStore, @ethio_commerce, @addis_market, @ethiopia_shopping
- **Batch processing** with progress tracking
- **Rate limiting** for API compliance
- **Database optimization** with indexing

### Entity Extraction
- **Prices**: Ethiopian Birr (ብር, ETB, birr) patterns
- **Products**: Amharic product names (ቦርሳ, ልብስ, ሞባይል, etc.)
- **Locations**: Ethiopian locations (አዲስ አበባ, ቦሌ, ገርጂ, etc.)

### Visualization Dashboards (6 Views)
1. **Channel Analysis** - Message distribution, views, trends
2. **Price Analysis** - Price ranges, inflation trends  
3. **Product & Location** - Entity frequency analysis
4. **Entity Correlation** - Performance metrics
5. **Time Series** - Multi-year activity patterns
6. **Summary Dashboard** - Key insights & metrics

## 🏷️ CoNLL Format Labeling (Task 2)

### BIO Tagging Scheme
- **B-PRODUCT**, **I-PRODUCT**: Product names (ቦርሳ, ሞባይል ፎን, cream)
- **B-LOC**, **I-LOC**: Ethiopian locations (አዲስ አበባ, ቦሌ, ገርጂ)
- **B-PRICE**, **I-PRICE**: Price expressions (ዋጋ 1200 ብር, በ 500 ETB)
- **O**: Outside any entity

### Labeling Statistics
- **50 messages labeled** (requirement: 30-50)
- **515 total tokens** processed
- **243 entity tokens** (47.2% coverage)
- **Entity distribution**: Product (47), Price (100), Location (40)

### Output Format Example
```
# Message 1: የሴቶች ቦርሳ ዋጋ 2500 ብር በአዲስ አበባ
የሴቶች	O
ቦርሳ	B-PRODUCT
ዋጋ	B-PRICE
2500	I-PRICE
ብር	I-PRICE
በአዲስ	O
አዲስ	B-LOC
አበባ	I-LOC
```

## 🔧 Configuration

### Environment Variables (.env)
```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE_NUMBER=your_phone_number
```

### Target Channels
- `@ShegerOnlineStore` - General e-commerce
- `@ethio_commerce` - Commercial products  
- `@addis_market` - Local marketplace
- `@ethiopia_shopping` - Shopping platform

## 📋 Task Progress

### ✅ Task 1: Data Collection & Processing
- [x] **Multi-channel Telegram scraping** (4+ channels)
- [x] **Amharic text processing** with regex patterns
- [x] **Entity extraction** (prices, products, locations)
- [x] **Database storage** with SQLite optimization
- [x] **Historical data collection** (2018-2025, 8 years)
- [x] **Comprehensive visualization** (6 dashboard views)
- [x] **Data export** for NER training preparation

**Outputs**:
- `data/historical_messages.db` - 8 years of historical data
- `data/processed/amharic_ecommerce.db` - Processed messages
- `notebook/06_data_visualization.ipynb` - Interactive dashboards
- `data/processed/historical_messages_2018_2025.csv` - Export file

### ✅ Task 2: CoNLL Format Labeling
- [x] **50 messages labeled** in CoNLL format (requirement: 30-50)
- [x] **BIO tagging scheme** (B-/I- for Product, Location, Price)
- [x] **Entity validation** with comprehensive metrics
- [x] **Plain text output** ready for NER training
- [x] **Quality assurance** with automated testing
- [x] **Format validation** ensuring proper BIO compliance

**Outputs**:
- `data/conll_labeled/amharic_ecommerce_conll.txt` - CoNLL format data
- `data/conll_labeled/labeling_statistics.json` - Detailed statistics
- `data/conll_labeled/validation_report.json` - Quality validation
- `notebook/Task2_CoNLL_Labeling.ipynb` - Interactive demonstration

### 🔄 Task 3: Model Development (Next)  
- [ ] **NER model fine-tuning** (XLM-Roberta, mBERT)
- [ ] **Model comparison** and evaluation
- [ ] **Performance metrics** (F1-score, precision, recall)
- [ ] **Model interpretability** (SHAP/LIME)
- [ ] **Production deployment** readiness

## 📊 Database Schema

### Historical Messages (Task 1)
```sql
CREATE TABLE historical_messages (
    id INTEGER,
    channel TEXT,
    channel_title TEXT,
    text TEXT,
    date TEXT,
    year INTEGER,
    month INTEGER,
    views INTEGER,
    has_media BOOLEAN,
    collection_timestamp TEXT,
    PRIMARY KEY (id, channel)
);
```

### Processed Messages (Task 1)
```sql
CREATE TABLE processed_messages (
    id INTEGER PRIMARY KEY,
    channel TEXT,
    channel_title TEXT,
    original_text TEXT,
    cleaned_text TEXT,
    date TEXT,
    views INTEGER,
    entities TEXT  -- JSON format
);
```

## 🔍 Data Analysis Features

### Task 1: Temporal Analysis (2018-2025)
- **Year-over-year growth** trends
- **Seasonal patterns** in e-commerce activity
- **Price inflation** tracking over 8 years
- **Market evolution** indicators

### Task 1: Entity Analytics
- **Product category** distribution and trends
- **Location-based** commerce patterns  
- **Price range** analysis by year/channel
- **Entity extraction** accuracy metrics

### Task 2: NER Training Data
- **CoNLL format compliance** validation
- **Entity coverage** analysis (47.2% of tokens)
- **BIO tagging** consistency checking
- **Label distribution** metrics

## 🛠️ Technical Stack

- **Python 3.8+** - Core language
- **Telethon** - Telegram API client
- **pandas** - Data manipulation
- **SQLite** - Database storage
- **matplotlib/seaborn** - Visualization
- **Jupyter** - Interactive development
- **asyncio** - Asynchronous processing
- **regex** - Amharic text pattern matching

## 📝 Usage Examples

### Task 1: Load Historical Data
```python
import pandas as pd
import sqlite3

# Load historical messages
with sqlite3.connect('data/historical_messages.db') as conn:
    df = pd.read_sql_query("SELECT * FROM historical_messages", conn)
    
print(f"Loaded {len(df)} messages from {df['year'].min()}-{df['year'].max()}")
```

### Task 1: Extract Entities
```python
from src.preprocessing.amharic_processor import AmharicTextProcessor

processor = AmharicTextProcessor()
entities = processor.extract_entities("ልብስ ዋጋ 500 ብር አዲስ አበባ")
# Returns: {'prices': ['500'], 'products': ['ልብስ'], 'locations': ['አዲስ አበባ']}
```

### Task 2: Load CoNLL Data
```python
# Load CoNLL format labeled data
with open('data/conll_labeled/amharic_ecommerce_conll.txt', 'r', encoding='utf-8') as f:
    conll_data = f.read()

# Parse tokens and labels
messages = []
current_message = []
for line in conll_data.split('\n'):
    if line.startswith('#') or not line.strip():
        if current_message:
            messages.append(current_message)
            current_message = []
    elif '\t' in line:
        token, label = line.strip().split('\t')
        current_message.append((token, label))

print(f"Loaded {len(messages)} labeled messages for NER training")
```

### Task 2: Validation Check
```python
import json

# Load validation report
with open('data/conll_labeled/validation_report.json', 'r') as f:
    report = json.load(f)

requirements = report['requirements_check']
print(f"Message count OK: {requirements['message_count_ok']}")
print(f"Format valid: {requirements['format_valid']}")
print(f"All entities found: {requirements['all_entity_types_found']}")
```

## 🎯 Project Deliverables

### Task 1 Deliverables
- ✅ **Multi-channel data collection** system
- ✅ **Historical database** (2018-2025)
- ✅ **Amharic text processing** pipeline
- ✅ **Visualization dashboards** (6 views)
- ✅ **Entity extraction** capabilities

### Task 2 Deliverables  
- ✅ **CoNLL format dataset** (50 messages)
- ✅ **BIO tagging validation** system
- ✅ **Quality metrics** and reporting
- ✅ **NER training ready** data format

### Ready for Task 3
- 🔄 **NER model training** pipeline
- 🔄 **Model evaluation** framework
- 🔄 **Performance benchmarking**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [MIT.md](MIT.md) file for details.

## 🔗 Links

- **Task 1 Documentation**: See [notebook/NOTEBOOKS_README.md](notebook/NOTEBOOKS_README.md) for detailed guide
- **Task 2 CoNLL Data**: See [data/conll_labeled/](data/conll_labeled/) for labeled training data
- **Interactive Notebooks**: Comprehensive inline documentation and examples
- **Historical Data**: Spans 2018-2025 for robust NER training and analysis