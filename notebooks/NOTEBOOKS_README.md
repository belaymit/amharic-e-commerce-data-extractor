# Amharic E-commerce Data Extractor Notebooks

This directory contains modular Jupyter notebooks for comprehensive data collection, processing, and analysis of Ethiopian e-commerce data from Telegram channels spanning **2018-2025**.

## 📁 Notebook Overview

### Core Data Collection (Sequential)
1. **`01_setup_and_config.ipynb`** - Environment setup and API configuration
2. **`02_amharic_processing.ipynb`** - Amharic text processing and entity extraction
3. **`03_multi_channel_scraping.ipynb`** - Multi-channel data collection (recent data)
4. **`04_data_processing.ipynb`** - Message processing and database storage

### Analysis and Visualization
5. **`05_export_and_analysis.ipynb`** - Data export and analysis with quick visualization
6. **`06_data_visualization.ipynb`** - Comprehensive visualization dashboards (6 different views)

### Historical Data Collection (NEW)
7. **`07_historical_data_collection.ipynb`** - **Large-scale historical data collection (2018-2025)**

## 🎯 Historical Data Collection (2018-2025)

The new **notebook #7** provides comprehensive historical data collection capabilities:

### Features:
- **7+ years of data**: Collection from 2018 to 2025
- **Batch processing**: Handles large datasets with progress tracking
- **Rate limiting**: Respects Telegram API limits
- **Database optimization**: Indexed database for performance
- **Year-wise analysis**: Distribution analysis across years
- **Sample data creation**: Demo version with realistic historical patterns

### Configuration:
```python
COLLECTION_CONFIG = {
    'start_year': 2018,
    'end_year': 2025,
    'max_messages_per_channel': 5000,
    'batch_size': 100,
    'target_channels': [
        "@ShegerOnlineStore",
        "@ethio_commerce", 
        "@addis_market",
        "@ethiopia_shopping"
    ]
}
```

### Database Structure:
- **`historical_messages.db`**: Main historical messages table
- **Indexed fields**: date, year, channel for fast queries  
- **Collection stats**: Per-channel statistics and year distributions
- **CSV export**: Ready for analysis in `historical_messages_2018_2025.csv`

## 🚀 Quick Start Guide

### For Historical Data Collection (2018-2025):
```bash
# 1. Run setup first
jupyter notebook 01_setup_and_config.ipynb

# 2. For comprehensive historical collection
jupyter notebook 07_historical_data_collection.ipynb

# 3. Visualize historical data
jupyter notebook 06_data_visualization.ipynb
```

### For Recent Data Collection:
```bash
# Sequential execution (recommended for first-time users)
jupyter notebook 01_setup_and_config.ipynb
jupyter notebook 02_amharic_processing.ipynb  
jupyter notebook 03_multi_channel_scraping.ipynb
jupyter notebook 04_data_processing.ipynb
jupyter notebook 05_export_and_analysis.ipynb
jupyter notebook 06_data_visualization.ipynb
```

## 📊 Visualization Capabilities

The visualization system automatically detects and uses historical data when available:

### Data Priority:
1. **Historical database** (2018-2025) - `../data/historical_messages.db`
2. **Processed database** - `../data/processed_messages.db`  
3. **Main database** - `../data/messages.db`
4. **Sample data** - Enhanced multi-year demo data

### Available Views:
1. **Channel Analysis** - Message distribution, views, time trends
2. **Price Analysis** - Price ranges, trends, channel comparison  
3. **Product & Location** - Entity distribution and mapping
4. **Entity Correlation** - Performance metrics and relationships
5. **Time Series** - Multi-year trends and patterns
6. **Summary Dashboard** - Key metrics and insights

## 🔧 Setup Requirements

### Environment Variables (.env):
```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash  
TELEGRAM_PHONE_NUMBER=your_phone_number
```

### Dependencies:
- Core: `telethon`, `pandas`, `sqlite3`
- Visualization: `matplotlib`, `seaborn`
- Processing: `asyncio`, `json`, `pathlib`

## 📈 Data Analysis Capabilities

With historical data (2018-2025), you can analyze:

### Temporal Trends:
- **Year-over-year growth** in e-commerce activity
- **Seasonal patterns** in product sales
- **Price evolution** and inflation trends
- **Channel performance** over time

### Entity Analysis:
- **Product category trends** across years
- **Location-based commerce patterns**
- **Price range distribution** by year/channel
- **Entity extraction accuracy** for NER training

### Business Intelligence:
- **Market growth indicators**
- **Consumer behavior evolution**  
- **Channel adoption patterns**
- **Regional e-commerce expansion**

## ⚠️ Important Notes

### Historical Data Collection:
- **Time intensive**: Full collection (2018-2025) may take several hours
- **API limits**: Respects Telegram rate limiting with delays
- **Storage**: Requires adequate disk space for large datasets
- **Demo mode**: Available for testing without full collection

### Execution Modes:
- **Independent**: Each notebook can run standalone
- **Sequential**: For complete pipeline execution
- **Selective**: Choose specific analysis components

## 🔍 Troubleshooting

### Historical Data Issues:
```python
# Check database status
verify_historical_database()

# Export for external analysis  
export_historical_data()

# Reset and recreate
setup_historical_database()
```

### Visualization Issues:
```python
# Check data loading priority
load_data()  # Shows which database is being used

# Force sample data creation
df = load_data()  # Creates multi-year sample if no DB found
```

### API Connection Issues:
```python
# Test credentials
test_telegram_connection()

# Check channel accessibility
test_channel_access(["@ShegerOnlineStore"])
```

## 📋 Next Steps

After completing data collection and visualization:

1. **Task 2**: CoNLL format labeling for NER training
2. **Model Training**: Use extracted entities for NER model development  
3. **Production Pipeline**: Deploy automated data collection system
4. **Business Intelligence**: Create automated reporting dashboards

The historical data (2018-2025) provides a comprehensive foundation for training robust NER models and understanding Ethiopian e-commerce evolution over time. 