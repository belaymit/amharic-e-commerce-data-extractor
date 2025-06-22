"""Data models for Telegram messages."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TelegramMessage(BaseModel):
    """Model for a Telegram message."""
    
    id: int
    channel_username: str
    channel_id: int
    text: Optional[str] = None
    date: datetime
    sender_id: Optional[int] = None
    views: Optional[int] = None
    forwards: Optional[int] = None
    replies: Optional[int] = None
    
    # Media information
    has_media: bool = False
    media_type: Optional[str] = None  # photo, video, document, etc.
    media_file_path: Optional[str] = None
    
    # Message metadata
    is_channel_post: bool = True
    is_reply: bool = False
    reply_to_message_id: Optional[int] = None
    
    # Raw data for debugging
    raw_data: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessedMessage(BaseModel):
    """Model for processed message data."""
    
    message_id: int
    channel_username: str
    original_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    normalized_text: Optional[str] = None
    tokens: Optional[List[str]] = None
    
    # Extracted metadata
    date: datetime
    views: Optional[int] = None
    
    # Processing flags
    is_processed: bool = False
    processing_timestamp: Optional[datetime] = None
    
    # Potential entities (for later NER processing)
    potential_products: Optional[List[str]] = None
    potential_prices: Optional[List[str]] = None
    potential_locations: Optional[List[str]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 