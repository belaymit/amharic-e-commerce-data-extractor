"""
Entity Extraction Module for Amharic E-commerce Text

Provides advanced entity extraction capabilities for products, prices, and locations
from Amharic e-commerce messages using pattern matching and NLP techniques.
"""

import re
from typing import Dict, List


class EntityExtractor:
    """Advanced entity extraction for Amharic e-commerce text analysis."""
    
    def __init__(self):
        """Initialize the entity extractor with predefined patterns and keywords."""
        self.price_patterns = [
            r'ዋጋ\s*(\d+)\s*ብር',          # "ዋጋ 1000 ብር"
            r'በ\s*(\d+)\s*ብር',            # "በ 500 ብር"
            r'(\d+)\s*ብር\s*ብቻ',          # "1200 ብር ብቻ"
            r'(\d+)\s*ብር(?!\w)',          # "800 ብር" (not followed by word)
            r'price\s*(\d+)',             # "price 1500"
            r'ETB\s*(\d+)',               # "ETB 2000"
            r'(\d{3,6})\s*birr?',         # "3000 birr"
        ]
        
        self.product_keywords = [
            # Clothing & Fashion
            'ቦርሳ', 'ጫማ', 'ልብስ', 'ሸሚዝ', 'ሱሪ', 'ጃኬት', 'አላባሽ', 'ቀሚስ',
            # Electronics & Tech
            'ሞባይል', 'ፎን', 'ቴሌፎን', 'laptop', 'computer', 'iPhone', 'Samsung',
            'case', 'charger', 'headphone', 'speaker', 'camera',
            # Beauty & Personal Care
            'ሻምፖ', 'soap', 'cream', 'lotion', 'perfume', 'makeup', 'lipstick',
            # Home & Kitchen
            'ሳህን', 'ኩባያ', 'መጠብ', 'ማንሻ', 'እቃዎች', 'furniture', 'table', 'chair',
            # General terms
            'collection', 'set', 'premium', 'quality', 'brand', 'original'
        ]
        
        self.location_keywords = [
            # Major areas in Addis Ababa
            'ቦሌ', 'መርካቶ', 'ፒያሳ', 'አዲስ', 'አበባ', 'ካዛንቺስ', 'ሰሚት',
            'ጂማ', 'ባህርዳር', 'ጎንደር', 'ሃዋሳ', 'ዲሬዳዋ', 'መከሌ',
            # Location indicators
            'ውስጥ', 'አካባቢ', 'ዞን', 'ክፍለ', 'ከተማ', 'አውራጃ',
            'around', 'area', 'zone', 'region'
        ]
    
    def extract_prices(self, text: str) -> List[int]:
        """Extract price information from text."""
        prices = []
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            prices.extend([int(match) for match in matches])
        return prices
    
    def extract_products(self, text: str) -> List[str]:
        """Extract product mentions from text."""
        products = []
        text_lower = text.lower()
        for product in self.product_keywords:
            if product.lower() in text_lower:
                products.append(product)
        return products
    
    def extract_locations(self, text: str) -> List[str]:
        """Extract location mentions from text."""
        locations = []
        text_lower = text.lower()
        for location in self.location_keywords:
            if location.lower() in text_lower:
                locations.append(location)
        return locations
    
    def extract_all_entities(self, text: str) -> Dict[str, List]:
        """
        Extract all entities (products, prices, locations) from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, List]: Dictionary containing extracted entities
        """
        return {
            'products': self.extract_products(text),
            'prices': self.extract_prices(text),
            'locations': self.extract_locations(text)
        }
    
    def get_price_statistics(self, prices: List[int]) -> Dict[str, float]:
        """Calculate price statistics from a list of prices."""
        if not prices:
            return {'avg': 0, 'min': 0, 'max': 0, 'range': 0, 'count': 0}
        
        return {
            'avg': sum(prices) / len(prices),
            'min': min(prices),
            'max': max(prices),
            'range': max(prices) - min(prices),
            'count': len(prices)
        }
    
    def classify_price_tier(self, avg_price: float) -> str:
        """Classify products into price tiers based on average price."""
        if avg_price >= 50000:
            return 'Premium'
        elif avg_price >= 10000:
            return 'Mid-range'
        elif avg_price >= 1000:
            return 'Budget'
        else:
            return 'Economy' 