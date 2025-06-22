"""
Data Loading Module for Vendor Analytics

Handles loading data from various sources with fallback mechanisms
and provides sample data generation for testing and demonstration.
"""

import pandas as pd
import sqlite3
import json
from typing import Tuple, Dict, List
from pathlib import Path

from .entity_extraction import EntityExtractor


class VendorDataLoader:
    """Manages data loading with fallback strategies and sample data generation."""
    
    def __init__(self):
        """Initialize the data loader with entity extractor."""
        self.entity_extractor = EntityExtractor()
    
    def load_from_database(self, db_path: str = "data/processed/amharic_ecommerce.db") -> Tuple[pd.DataFrame, bool]:
        """
        Load vendor data from SQLite database.
        
        Args:
            db_path (str): Path to the database file
            
        Returns:
            Tuple[pd.DataFrame, bool]: DataFrame and success flag
        """
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM processed_messages", conn)
            conn.close()
            
            if len(df) > 0:
                print(f"✅ SUCCESS: Loaded {len(df)} messages from database")
                return df, True
        except Exception as e:
            print(f"⚠️  Database error: {str(e)[:100]}...")
        
        return pd.DataFrame(), False
    
    def create_sample_vendor_profiles(self) -> Dict:
        """Create comprehensive sample vendor profiles for testing."""
        return {
            '@EthioFashionHub': {
                'title': 'Ethiopian Fashion Hub',
                'category': 'Fashion & Clothing',
                'business_model': 'Fashion Retailer',
                'target_market': 'Mid-range',
                'messages': [
                    {'text': 'የሴቶች ቦርሳ በጣም ጥሩ ዋጋ 2500 ብር ቦሌ ውስጥ ይገኛል', 'views': 1850, 'date': '2025-01-15'},
                    {'text': 'ጫማ collection ዋጋ 3000 ብር የተለያዩ ቀለም አዲስ አበባ', 'views': 1200, 'date': '2025-01-14'},
                    {'text': 'የወንዶች ሸሚዝ premium quality ዋጋ 1800 ብር', 'views': 980, 'date': '2025-01-13'},
                    {'text': 'ልብስ ሽያጭ በ 1200 ብር ፒያሳ አካባቢ', 'views': 1350, 'date': '2025-01-12'},
                    {'text': 'አላባሽ collection ዋጋ 2200 ብር ቦሌ', 'views': 1100, 'date': '2025-01-11'},
                    {'text': 'ሱሪ እና ሸሚዝ set ዋጋ 2800 ብር', 'views': 1400, 'date': '2025-01-10'},
                    {'text': 'የሴቶች ጫማ ዋጋ 2000 ብር መርካቶ ውስጥ', 'views': 1050, 'date': '2025-01-09'},
                    {'text': 'ጃኬት winter collection 3500 ብር', 'views': 1600, 'date': '2025-01-08'},
                ]
            },
            '@AddisElectronics': {
                'title': 'Addis Electronics Center',
                'category': 'Electronics & Technology',
                'business_model': 'Electronics Retailer',
                'target_market': 'High-end',
                'messages': [
                    {'text': 'ሞባይል ፎን Samsung Galaxy ዋጋ 15000 ብር ፒያሳ', 'views': 4200, 'date': '2025-01-15'},
                    {'text': 'laptop HP በ 45000 ብር original warranty ቦሌ', 'views': 3800, 'date': '2025-01-14'},
                    {'text': 'iPhone 13 case ዋጋ 800 ብር አዲስ አበባ', 'views': 2100, 'date': '2025-01-13'},
                    {'text': 'ሞባይል charger original በ 600 ብር', 'views': 1800, 'date': '2025-01-12'},
                    {'text': 'headphone Bluetooth ዋጋ 2500 ብር', 'views': 2200, 'date': '2025-01-11'},
                    {'text': 'camera digital በ 25000 ብር professional', 'views': 3200, 'date': '2025-01-10'},
                ]
            },
            '@MerkatoGeneralStore': {
                'title': 'Merkato General Market',
                'category': 'General Merchandise',
                'business_model': 'General Retailer',
                'target_market': 'Budget-friendly',
                'messages': [
                    {'text': 'ሻምፖ እና soap set ዋጋ 450 ብር መርካቶ', 'views': 680, 'date': '2025-01-15'},
                    {'text': 'የቤት እቃዎች collection በ 2800 ብር', 'views': 520, 'date': '2025-01-14'},
                    {'text': 'ኩባያ እና ሳህን set 800 ብር', 'views': 420, 'date': '2025-01-13'},
                    {'text': 'cream እና lotion ዋጋ 350 ብር', 'views': 380, 'date': '2025-01-12'},
                ]
            },
            '@BolePremiumShopping': {
                'title': 'Bole Premium Shopping Center',
                'category': 'Premium Products',
                'business_model': 'Luxury Retailer',
                'target_market': 'Premium',
                'messages': [
                    {'text': 'laptop MacBook Pro ዋጋ 85000 ብር original ቦሌ', 'views': 5200, 'date': '2025-01-15'},
                    {'text': 'iPhone 14 Pro Max በ 55000 ብር warranty', 'views': 4800, 'date': '2025-01-14'},
                    {'text': 'የሴቶች ቦርሳ luxury brand 8500 ብር', 'views': 3200, 'date': '2025-01-13'},
                    {'text': 'ጫማ Nike original collection 12000 ብር', 'views': 3800, 'date': '2025-01-12'},
                    {'text': 'perfume premium brand ዋጋ 4500 ብር', 'views': 2900, 'date': '2025-01-11'},
                    {'text': 'camera Sony professional 95000 ብር', 'views': 4100, 'date': '2025-01-10'},
                    {'text': 'laptop gaming setup 120000 ብር complete', 'views': 5800, 'date': '2025-01-09'},
                ]
            },
            '@HawassaBeautyShop': {
                'title': 'Hawassa Beauty & Care',
                'category': 'Beauty & Personal Care',
                'business_model': 'Beauty Specialist',
                'target_market': 'Mid-range',
                'messages': [
                    {'text': 'makeup collection premium ዋጋ 3200 ብር ሃዋሳ', 'views': 1400, 'date': '2025-01-15'},
                    {'text': 'ሻምፖ organic በ 680 ብር natural', 'views': 920, 'date': '2025-01-14'},
                    {'text': 'lipstick set ዋጋ 1200 ብር የተለያዩ ቀለም', 'views': 1100, 'date': '2025-01-13'},
                    {'text': 'cream anti-aging 2800 ብር imported', 'views': 1300, 'date': '2025-01-12'},
                ]
            }
        }
    
    def generate_sample_dataframe(self) -> pd.DataFrame:
        """Generate a comprehensive sample DataFrame for testing."""
        vendor_profiles = self.create_sample_vendor_profiles()
        
        all_messages = []
        for channel, profile in vendor_profiles.items():
            for msg in profile['messages']:
                entities = self.entity_extractor.extract_all_entities(msg['text'])
                all_messages.append({
                    'channel': channel,
                    'channel_title': profile['title'],
                    'category': profile['category'],
                    'business_model': profile['business_model'],
                    'target_market': profile['target_market'],
                    'text': msg['text'],
                    'views': msg['views'],
                    'date': msg['date'],
                    'entities': json.dumps(entities)
                })
        
        return pd.DataFrame(all_messages)
    
    def load_vendor_data(self, db_path: str = "data/processed/amharic_ecommerce.db") -> Tuple[pd.DataFrame, str]:
        """
        Load vendor data with fallback strategy.
        
        Args:
            db_path (str): Path to database file
            
        Returns:
            Tuple[pd.DataFrame, str]: DataFrame and data source type
        """
        print("🔄 Loading vendor data...")
        
        # Try loading real data first
        df, success = self.load_from_database(db_path)
        if success:
            print(f"📊 Real data: {len(df)} messages from {df['channel'].nunique()} vendors")
            return df, "real"
        
        # Fall back to sample data
        print("📊 DEMO MODE: Creating comprehensive sample data")
        df = self.generate_sample_dataframe()
        print(f"✨ Sample data: {len(df)} messages from {df['channel'].nunique()} vendors")
        return df, "sample"
    
    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the required columns.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['channel', 'channel_title', 'text', 'views', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ Data structure validation passed")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a summary of the loaded data."""
        return {
            'total_messages': len(df),
            'unique_vendors': df['channel'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'categories': df['category'].unique().tolist() if 'category' in df.columns else [],
            'avg_views': df['views'].mean() if 'views' in df.columns else 0
        } 