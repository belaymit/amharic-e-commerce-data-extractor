"""Telegram scraper for extracting messages from Ethiopian e-commerce channels."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, AsyncGenerator, Dict, Any
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, UsernameNotOccupiedError
from telethon.tl.types import Channel, Chat
from loguru import logger

from ..models import TelegramMessage
from ..config import telegram_config, data_config


class TelegramScraper:
    """Telegram scraper for e-commerce channels."""
    
    def __init__(self):
        """Initialize the Telegram scraper."""
        self.client: Optional[TelegramClient] = None
        self._session_file = data_config.base_data_dir / f"{telegram_config.session_name}.session"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Connect to Telegram API."""
        if not telegram_config.api_id or not telegram_config.api_hash:
            raise ValueError("Telegram API credentials not provided. Please set TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables.")
        
        self.client = TelegramClient(
            str(self._session_file),
            telegram_config.api_id,
            telegram_config.api_hash
        )
        
        await self.client.start(phone=telegram_config.phone_number)
        logger.info("Connected to Telegram API")
    
    async def disconnect(self) -> None:
        """Disconnect from Telegram API."""
        if self.client:
            await self.client.disconnect()
            logger.info("Disconnected from Telegram API")
    
    async def get_channel_info(self, channel_username: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a channel."""
        try:
            entity = await self.client.get_entity(channel_username)
            
            if isinstance(entity, (Channel, Chat)):
                return {
                    "id": entity.id,
                    "title": entity.title,
                    "username": getattr(entity, 'username', None),
                    "participants_count": getattr(entity, 'participants_count', None),
                    "is_channel": isinstance(entity, Channel)
                }
        except (ChannelPrivateError, UsernameNotOccupiedError) as e:
            logger.warning(f"Cannot access channel {channel_username}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_username}: {e}")
            return None
    
    async def scrape_channel_messages(
        self,
        channel_username: str,
        limit: int = 1000,
        offset_date: Optional[datetime] = None
    ) -> AsyncGenerator[TelegramMessage, None]:
        """Scrape messages from a specific channel."""
        if not self.client:
            raise RuntimeError("Client not connected. Use async context manager or call connect() first.")
        
        try:
            entity = await self.client.get_entity(channel_username)
            logger.info(f"Starting to scrape messages from {channel_username}")
            
            message_count = 0
            async for message in self.client.iter_messages(
                entity,
                limit=limit,
                offset_date=offset_date,
                reverse=False
            ):
                if message.text or message.media:
                    telegram_message = self._convert_to_telegram_message(message, channel_username)
                    yield telegram_message
                    message_count += 1
                    
                    if message_count % 100 == 0:
                        logger.info(f"Scraped {message_count} messages from {channel_username}")
            
            logger.info(f"Completed scraping {message_count} messages from {channel_username}")
            
        except (ChannelPrivateError, UsernameNotOccupiedError) as e:
            logger.warning(f"Cannot access channel {channel_username}: {e}")
        except Exception as e:
            logger.error(f"Error scraping channel {channel_username}: {e}")
    
    def _convert_to_telegram_message(self, message, channel_username: str) -> TelegramMessage:
        """Convert Telethon message to TelegramMessage model."""
        return TelegramMessage(
            id=message.id,
            channel_username=channel_username,
            channel_id=message.peer_id.channel_id if hasattr(message.peer_id, 'channel_id') else 0,
            text=message.text,
            date=message.date,
            sender_id=message.sender_id,
            views=getattr(message, 'views', None),
            forwards=getattr(message, 'forwards', None),
            replies=getattr(message.replies, 'replies', None) if message.replies else None,
            has_media=bool(message.media),
            media_type=type(message.media).__name__ if message.media else None,
            is_channel_post=not message.out,
            is_reply=bool(message.reply_to),
            reply_to_message_id=message.reply_to.reply_to_msg_id if message.reply_to else None,
            raw_data={
                "message_id": message.id,
                "date": message.date.isoformat(),
                "peer_id": str(message.peer_id) if message.peer_id else None
            }
        )
    
    async def scrape_multiple_channels(
        self,
        channel_usernames: List[str],
        limit_per_channel: int = 1000,
        offset_date: Optional[datetime] = None
    ) -> AsyncGenerator[TelegramMessage, None]:
        """Scrape messages from multiple channels."""
        for channel_username in channel_usernames:
            logger.info(f"Processing channel: {channel_username}")
            
            # Get channel info first
            channel_info = await self.get_channel_info(channel_username)
            if not channel_info:
                logger.warning(f"Skipping inaccessible channel: {channel_username}")
                continue
            
            # Scrape messages
            async for message in self.scrape_channel_messages(
                channel_username,
                limit=limit_per_channel,
                offset_date=offset_date
            ):
                yield message
            
            # Small delay between channels to be respectful
            await asyncio.sleep(1) 