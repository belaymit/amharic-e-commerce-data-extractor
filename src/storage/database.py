"""Database handler for storing Telegram messages and processed data."""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
from loguru import logger

from ..models import TelegramMessage, ProcessedMessage
from ..config import data_config


class DatabaseHandler:
    """Handle database operations for Telegram messages and processed data."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database handler."""
        self.db_path = db_path or data_config.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Raw messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telegram_messages (
                    id INTEGER,
                    channel_username TEXT,
                    channel_id INTEGER,
                    text TEXT,
                    date TIMESTAMP,
                    sender_id INTEGER,
                    views INTEGER,
                    forwards INTEGER,
                    replies INTEGER,
                    has_media BOOLEAN,
                    media_type TEXT,
                    media_file_path TEXT,
                    is_channel_post BOOLEAN,
                    is_reply BOOLEAN,
                    reply_to_message_id INTEGER,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id, channel_username)
                )
            """)
            
            # Processed messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id INTEGER,
                    channel_username TEXT,
                    original_text TEXT,
                    cleaned_text TEXT,
                    normalized_text TEXT,
                    tokens TEXT,
                    date TIMESTAMP,
                    views INTEGER,
                    is_processed BOOLEAN,
                    processing_timestamp TIMESTAMP,
                    potential_products TEXT,
                    potential_prices TEXT,
                    potential_locations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (message_id, channel_username),
                    FOREIGN KEY (message_id, channel_username) 
                        REFERENCES telegram_messages(id, channel_username)
                )
            """)
            
            # Channel info table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS channel_info (
                    username TEXT PRIMARY KEY,
                    channel_id INTEGER,
                    title TEXT,
                    participants_count INTEGER,
                    is_channel BOOLEAN,
                    last_scraped TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def save_telegram_message(self, message: TelegramMessage) -> bool:
        """Save a Telegram message to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO telegram_messages 
                    (id, channel_username, channel_id, text, date, sender_id, views, 
                     forwards, replies, has_media, media_type, media_file_path, 
                     is_channel_post, is_reply, reply_to_message_id, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.channel_username,
                    message.channel_id,
                    message.text,
                    message.date,
                    message.sender_id,
                    message.views,
                    message.forwards,
                    message.replies,
                    message.has_media,
                    message.media_type,
                    message.media_file_path,
                    message.is_channel_post,
                    message.is_reply,
                    message.reply_to_message_id,
                    json.dumps(message.raw_data) if message.raw_data else None
                ))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving message {message.id}: {e}")
            return False
    
    def save_processed_message(self, message: ProcessedMessage) -> bool:
        """Save a processed message to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_messages 
                    (message_id, channel_username, original_text, cleaned_text, 
                     normalized_text, tokens, date, views, is_processed, 
                     processing_timestamp, potential_products, potential_prices, 
                     potential_locations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.message_id,
                    message.channel_username,
                    message.original_text,
                    message.cleaned_text,
                    message.normalized_text,
                    json.dumps(message.tokens) if message.tokens else None,
                    message.date,
                    message.views,
                    message.is_processed,
                    message.processing_timestamp,
                    json.dumps(message.potential_products) if message.potential_products else None,
                    json.dumps(message.potential_prices) if message.potential_prices else None,
                    json.dumps(message.potential_locations) if message.potential_locations else None
                ))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving processed message {message.message_id}: {e}")
            return False
    
    def save_channel_info(self, username: str, info: Dict[str, Any]) -> bool:
        """Save channel information to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO channel_info 
                    (username, channel_id, title, participants_count, is_channel, last_scraped)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    username,
                    info.get('id'),
                    info.get('title'),
                    info.get('participants_count'),
                    info.get('is_channel', False),
                    datetime.now()
                ))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving channel info for {username}: {e}")
            return False
    
    def get_messages_by_channel(self, channel_username: str, limit: Optional[int] = None) -> List[Dict]:
        """Get messages from a specific channel."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM telegram_messages 
                    WHERE channel_username = ? 
                    ORDER BY date DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                return pd.read_sql_query(query, conn, params=(channel_username,)).to_dict('records')
        except Exception as e:
            logger.error(f"Error getting messages for {channel_username}: {e}")
            return []
    
    def get_processed_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """Get processed messages."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM processed_messages 
                    WHERE is_processed = 1 
                    ORDER BY processing_timestamp DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                return pd.read_sql_query(query, conn).to_dict('records')
        except Exception as e:
            logger.error(f"Error getting processed messages: {e}")
            return []
    
    def get_channel_stats(self) -> Dict[str, Any]:
        """Get statistics about scraped channels."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        channel_username,
                        COUNT(*) as message_count,
                        AVG(views) as avg_views,
                        MIN(date) as first_message,
                        MAX(date) as last_message
                    FROM telegram_messages 
                    GROUP BY channel_username
                """
                
                stats = pd.read_sql_query(query, conn).to_dict('records')
                return {stat['channel_username']: stat for stat in stats}
        except Exception as e:
            logger.error(f"Error getting channel stats: {e}")
            return {}
    
    def export_to_csv(self, table_name: str, output_path: Optional[Path] = None) -> bool:
        """Export table data to CSV."""
        try:
            if output_path is None:
                output_path = data_config.processed_data_dir / f"{table_name}.csv"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                df.to_csv(output_path, index=False)
                logger.info(f"Exported {table_name} to {output_path}")
                return True
        except Exception as e:
            logger.error(f"Error exporting {table_name}: {e}")
            return False
    
    def batch_save_messages(self, messages: List[TelegramMessage]) -> int:
        """Save multiple messages in batch."""
        saved_count = 0
        for message in messages:
            if self.save_telegram_message(message):
                saved_count += 1
        
        logger.info(f"Saved {saved_count}/{len(messages)} messages")
        return saved_count
    
    def batch_save_processed_messages(self, messages: List[ProcessedMessage]) -> int:
        """Save multiple processed messages in batch."""
        saved_count = 0
        for message in messages:
            if self.save_processed_message(message):
                saved_count += 1
        
        logger.info(f"Saved {saved_count}/{len(messages)} processed messages")
        return saved_count 