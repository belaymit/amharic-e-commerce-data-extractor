# Amharic E-commerce Data Extractor

A comprehensive system for collecting, processing, and analyzing Ethiopian e-commerce data from Telegram channels with **historical data collection spanning 2018-2025**.

## 🎯 Project Overview

This project extracts entities (products, prices, locations) from Amharic text in Ethiopian e-commerce Telegram channels to create training data for Named Entity Recognition (NER) models.

### Key Features
- **Historical Data Collection**: 2018-2025 comprehensive scraping
- **Multi-channel Support**: 4+ Ethiopian e-commerce channels
- **Amharic Text Processing**: Entity extraction with regex patterns
- **Database Storage**: SQLite with optimized indexing
- **Data Visualization**: 6 different dashboard views
- **NER Training Ready**: Structured entity extraction

## 📁 Project Structure

```
amharic-e-commerce-data-extractor/
├── notebook/                           # Jupyter notebooks (Task 1)
│   ├── 01_setup_and_config.ipynb      # Environment setup
│   ├── 02_amharic_processing.ipynb    # Text processing
│   ├── 03_multi_channel_scraping.ipynb # Recent data collection
│   ├── 04_data_processing.ipynb       # Message processing
│   ├── 05_export_and_analysis.ipynb   # Data export & analysis
│   ├── 06_data_visualization.ipynb    # Comprehensive visualization
│   ├── 07_historical_data_collection.ipynb # 2018-2025 collection
│   └── NOTEBOOKS_README.md            # Detailed notebook guide
├── src/                               # Source code modules
│   ├── config/settings.py            # Configuration management
│   ├── scrapers/telegram_scraper.py  # Telegram scraping
│   ├── preprocessing/amharic_processor.py # Text processing
│   ├── pipeline/data_ingestion.py    # Data pipeline
│   └── storage/database.py           # Database operations
├── data/                             # Data storage
│   ├── raw/                         # Raw scraped data
│   ├── processed/                   # Processed datasets
│   └── historical_messages.db       # Historical data (2018-2025)
├── requirements.txt                  # Python dependencies
├── setup.py                         # Package setup
├── run_task1.py                     # Main execution script
└── .env.example                     # Environment variables template
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

### 2. Historical Data Collection (2018-2025)
```bash
# Run historical data collection
jupyter notebook notebook/07_historical_data_collection.ipynb

# Visualize historical data
jupyter notebook notebook/06_data_visualization.ipynb
```

### 3. Recent Data Collection
```bash
# Sequential execution for complete pipeline
jupyter notebook notebook/01_setup_and_config.ipynb
jupyter notebook notebook/02_amharic_processing.ipynb
jupyter notebook notebook/03_multi_channel_scraping.ipynb
jupyter notebook notebook/04_data_processing.ipynb
jupyter notebook notebook/05_export_and_analysis.ipynb
```

## 📊 Data Collection Capabilities

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

## 📈 Visualization Dashboards

6 comprehensive visualization views:
1. **Channel Analysis** - Message distribution, views, trends
2. **Price Analysis** - Price ranges, inflation trends  
3. **Product & Location** - Entity frequency analysis
4. **Entity Correlation** - Performance metrics
5. **Time Series** - Multi-year activity patterns
6. **Summary Dashboard** - Key insights & metrics

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
- [x] Multi-channel Telegram scraping
- [x] Amharic text processing
- [x] Entity extraction (prices, products, locations)
- [x] Database storage with SQLite
- [x] Historical data collection (2018-2025)
- [x] Comprehensive visualization system
- [x] Data export for NER training

### 🔄 Task 2: NER Training Data (Next)
- [ ] CoNLL format labeling
- [ ] Entity annotation for training
- [ ] Train/validation/test splits

### 🔄 Task 3: Model Development (Future)  
- [ ] NER model training
- [ ] Model evaluation & tuning
- [ ] Production deployment

## 📊 Database Schema

### Historical Messages
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

## 🔍 Data Analysis Features

### Temporal Analysis (2018-2025)
- **Year-over-year growth** trends
- **Seasonal patterns** in e-commerce activity
- **Price inflation** tracking over 8 years
- **Market evolution** indicators

### Entity Analytics
- **Product category** distribution and trends
- **Location-based** commerce patterns  
- **Price range** analysis by year/channel
- **Entity extraction** accuracy metrics

## 🛠️ Technical Stack

- **Python 3.8+** - Core language
- **Telethon** - Telegram API client
- **pandas** - Data manipulation
- **SQLite** - Database storage
- **matplotlib/seaborn** - Visualization
- **Jupyter** - Interactive development
- **asyncio** - Asynchronous processing

## 📝 Usage Examples

### Load Historical Data
```python
import pandas as pd
import sqlite3

# Load historical messages
with sqlite3.connect('data/historical_messages.db') as conn:
    df = pd.read_sql_query("SELECT * FROM historical_messages", conn)
    
print(f"Loaded {len(df)} messages from {df['year'].min()}-{df['year'].max()}")
```

### Extract Entities
```python
from src.preprocessing.amharic_processor import AmharicProcessor

processor = AmharicProcessor()
entities = processor.extract_entities("ልብስ ዋጋ 500 ብር አዲስ አበባ")
# Returns: {'prices': ['500'], 'products': ['ልብስ'], 'locations': ['አዲስ አበባ']}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [MIT.md](MIT.md) file for details.

## 🔗 Links

- **Notebooks**: See [notebook/NOTEBOOKS_README.md](notebook/NOTEBOOKS_README.md) for detailed guide
- **Documentation**: Comprehensive inline documentation in notebooks
- **Data**: Historical data spans 2018-2025 for robust NER training