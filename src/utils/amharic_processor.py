"""Amharic text preprocessing utilities."""

import re
from typing import List, Optional, Dict, Any
from datetime import datetime

import regex
from loguru import logger

from ..models import ProcessedMessage, TelegramMessage


class AmharicTextProcessor:
    """Processor for Amharic text data."""
    
    def __init__(self):
        """Initialize the Amharic text processor."""
        self._setup_patterns()
    
    def _setup_patterns(self) -> None:
        """Setup regex patterns for Amharic text processing."""
        # Amharic unicode range
        self.amharic_pattern = regex.compile(r'[\u1200-\u137F]+')
        
        # Price patterns (Ethiopian Birr)
        self.price_patterns = [
            regex.compile(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr|ETB)', regex.IGNORECASE),
            regex.compile(r'(?:ዋጋ|ዋጋው|በ)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', regex.IGNORECASE),
            regex.compile(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር)', regex.IGNORECASE)
        ]
        
        # Location patterns (Ethiopian cities/locations)
        self.location_keywords = [
            'አዲስ አበባ', 'አዲስ ዓባባ', 'ቦሌ', 'ቢሽፍቱ', 'ጎንደር', 'ባህር ዳር', 'ሐዋሳ', 'ንግሥት', 'መቀሌ',
            'ዲሬዳዋ', 'ካሳንቺስ', 'ገርጂ', 'ሰሚት', 'ላጋጣፎ', 'ሰንጋቴ', 'ሰሚት', 'ማርካቶ', 'ፒያሳ'
        ]
        
        # URL pattern
        self.url_pattern = regex.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Telegram username pattern
        self.username_pattern = regex.compile(r'@[a-zA-Z0-9_]+')
        
        # Phone number pattern (Ethiopian format)
        self.phone_pattern = regex.compile(r'(?:\+251|0)\d{9}')
        
        # Emoji pattern
        self.emoji_pattern = regex.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing."""
        if not text:
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove excessive whitespace
        text = regex.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_amharic_text(self, text: str) -> str:
        """Normalize Amharic text by handling character variations."""
        if not text:
            return ""
        
        # Common Amharic character normalizations
        normalizations = {
            'ሀ': 'ሃ',  # Normalize ha
            'ሐ': 'ሃ',  # Normalize ha
            'ሸ': 'ሽ',  # Normalize sha
            'ጸ': 'ፀ',  # Normalize tsa
        }
        
        normalized_text = text
        for old_char, new_char in normalizations.items():
            normalized_text = normalized_text.replace(old_char, new_char)
        
        return normalized_text
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text into words."""
        if not text:
            return []
        
        # Simple word-based tokenization
        # Split on whitespace and punctuation, but preserve Amharic characters
        tokens = regex.findall(r'[\u1200-\u137F]+|[a-zA-Z0-9]+', text)
        
        # Filter out empty tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def extract_potential_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities from text."""
        entities = {
            'prices': [],
            'locations': [],
            'products': [],
            'contacts': []
        }
        
        if not text:
            return entities
        
        # Extract prices
        for price_pattern in self.price_patterns:
            matches = price_pattern.findall(text)
            entities['prices'].extend(matches)
        
        # Extract locations
        for location in self.location_keywords:
            if location in text:
                entities['locations'].append(location)
        
        # Extract phone numbers
        phone_matches = self.phone_pattern.findall(text)
        entities['contacts'].extend(phone_matches)
        
        # Extract usernames
        username_matches = self.username_pattern.findall(text)
        entities['contacts'].extend(username_matches)
        
        # Simple product extraction (words that appear with prices)
        price_context_pattern = regex.compile(r'([^\s]+(?:\s+[^\s]+){0,2})\s*(?:ዋጋ|በ|\d+\s*ብር)', regex.IGNORECASE)
        product_matches = price_context_pattern.findall(text)
        entities['products'].extend(product_matches)
        
        # Remove duplicates and clean
        for key in entities:
            entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
        
        return entities
    
    def process_message(self, message: TelegramMessage) -> ProcessedMessage:
        """Process a single Telegram message."""
        if not message.text:
            return ProcessedMessage(
                message_id=message.id,
                channel_username=message.channel_username,
                date=message.date,
                views=message.views,
                is_processed=False,
                processing_timestamp=datetime.now()
            )
        
        # Clean and normalize text
        cleaned_text = self.clean_text(message.text)
        normalized_text = self.normalize_amharic_text(cleaned_text)
        
        # Tokenize
        tokens = self.tokenize_amharic(normalized_text)
        
        # Extract potential entities
        entities = self.extract_potential_entities(normalized_text)
        
        return ProcessedMessage(
            message_id=message.id,
            channel_username=message.channel_username,
            original_text=message.text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            tokens=tokens,
            date=message.date,
            views=message.views,
            is_processed=True,
            processing_timestamp=datetime.now(),
            potential_products=entities['products'],
            potential_prices=entities['prices'],
            potential_locations=entities['locations']
        )
    
    def process_batch(self, messages: List[TelegramMessage]) -> List[ProcessedMessage]:
        """Process a batch of messages."""
        processed_messages = []
        
        for message in messages:
            try:
                processed_message = self.process_message(message)
                processed_messages.append(processed_message)
            except Exception as e:
                logger.error(f"Error processing message {message.id}: {e}")
                # Create a minimal processed message for failed cases
                processed_messages.append(ProcessedMessage(
                    message_id=message.id,
                    channel_username=message.channel_username,
                    original_text=message.text,
                    date=message.date,
                    views=message.views,
                    is_processed=False,
                    processing_timestamp=datetime.now()
                ))
        
        logger.info(f"Processed {len(processed_messages)} messages")
        return processed_messages 