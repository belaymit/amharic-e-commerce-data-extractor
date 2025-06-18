"""Data ingestion pipeline for Telegram e-commerce channels."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from loguru import logger

from ..config import app_config, data_config
from ..scrapers import TelegramScraper
from ..preprocessing import AmharicTextProcessor
from ..storage import DatabaseHandler
from ..models import TelegramMessage


class DataIngestionPipeline:
    """Main pipeline for data ingestion and preprocessing."""
    
    def __init__(self):
        """Initialize the data ingestion pipeline."""
        self.scraper = TelegramScraper()
        self.processor = AmharicTextProcessor()
        self.db_handler = DatabaseHandler()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logger.add(
            app_config.log_file,
            rotation="10 MB",
            retention="7 days",
            level=app_config.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
        )
    
    async def run_ingestion(
        self,
        channels: Optional[List[str]] = None,
        limit_per_channel: int = 1000,
        days_back: int = 30
    ) -> dict:
        """Run the complete data ingestion pipeline."""
        channels = channels or app_config.target_channels
        offset_date = datetime.now() - timedelta(days=days_back)
        
        logger.info(f"Starting data ingestion for {len(channels)} channels")
        logger.info(f"Channels: {', '.join(channels)}")
        logger.info(f"Limit per channel: {limit_per_channel}")
        logger.info(f"Going back {days_back} days from {offset_date}")
        
        stats = {
            'total_messages_scraped': 0,
            'total_messages_processed': 0,
            'channels_processed': 0,
            'errors': [],
            'channel_stats': {}
        }
        
        try:
            async with self.scraper:
                await self._process_channels(channels, limit_per_channel, offset_date, stats)
            
            logger.info("Data ingestion completed successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            stats['errors'].append(str(e))
            return stats
    
    async def _process_channels(
        self,
        channels: List[str],
        limit_per_channel: int,
        offset_date: datetime,
        stats: dict
    ) -> None:
        """Process all channels."""
        for channel in channels:
            try:
                channel_stats = await self._process_single_channel(
                    channel, limit_per_channel, offset_date
                )
                
                stats['channels_processed'] += 1
                stats['total_messages_scraped'] += channel_stats['scraped']
                stats['total_messages_processed'] += channel_stats['processed']
                stats['channel_stats'][channel] = channel_stats
                
                logger.info(f"Completed processing {channel}: {channel_stats}")
                
            except Exception as e:
                error_msg = f"Error processing channel {channel}: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
    
    async def _process_single_channel(
        self,
        channel: str,
        limit: int,
        offset_date: datetime
    ) -> dict:
        """Process a single channel."""
        logger.info(f"Processing channel: {channel}")
        
        # Get channel info and save it
        channel_info = await self.scraper.get_channel_info(channel)
        if channel_info:
            self.db_handler.save_channel_info(channel, channel_info)
        
        # Collect messages in batches
        scraped_count = 0
        processed_count = 0
        message_batch = []
        
        async for message in self.scraper.scrape_channel_messages(
            channel, limit=limit, offset_date=offset_date
        ):
            message_batch.append(message)
            scraped_count += 1
            
            # Process in batches
            if len(message_batch) >= data_config.batch_size:
                batch_processed = await self._process_message_batch(message_batch)
                processed_count += batch_processed
                message_batch = []
                
                logger.info(f"Processed batch for {channel}: {processed_count} total processed")
        
        # Process remaining messages
        if message_batch:
            batch_processed = await self._process_message_batch(message_batch)
            processed_count += batch_processed
        
        return {
            'scraped': scraped_count,
            'processed': processed_count,
            'channel_info': channel_info
        }
    
    async def _process_message_batch(self, messages: List[TelegramMessage]) -> int:
        """Process a batch of messages."""
        # Save raw messages to database
        saved_count = self.db_handler.batch_save_messages(messages)
        
        # Process messages for NER
        processed_messages = self.processor.process_batch(messages)
        
        # Save processed messages
        self.db_handler.batch_save_processed_messages(processed_messages)
        
        return len(processed_messages)
    
    def export_data(self, format: str = 'csv') -> bool:
        """Export scraped and processed data."""
        try:
            if format.lower() == 'csv':
                # Export all tables to CSV
                tables = ['telegram_messages', 'processed_messages', 'channel_info']
                for table in tables:
                    self.db_handler.export_to_csv(table)
                
                logger.info("Data exported successfully")
                return True
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def get_ingestion_summary(self) -> dict:
        """Get summary of ingested data."""
        try:
            channel_stats = self.db_handler.get_channel_stats()
            
            total_messages = sum(stats['message_count'] for stats in channel_stats.values())
            total_channels = len(channel_stats)
            
            summary = {
                'total_channels': total_channels,
                'total_messages': total_messages,
                'channel_breakdown': channel_stats,
                'database_path': str(self.db_handler.db_path)
            }
            
            logger.info(f"Ingestion summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting ingestion summary: {e}")
            return {}


async def run_task_1(
    channels: Optional[List[str]] = None,
    limit_per_channel: int = 1000,
    days_back: int = 30,
    export_csv: bool = True
) -> dict:
    """
    Main function to run Task 1: Data Ingestion and Preprocessing.
    
    Args:
        channels: List of Telegram channels to scrape
        limit_per_channel: Maximum messages per channel
        days_back: How many days back to scrape
        export_csv: Whether to export data to CSV
    
    Returns:
        Dictionary with ingestion statistics
    """
    pipeline = DataIngestionPipeline()
    
    # Run the ingestion
    stats = await pipeline.run_ingestion(
        channels=channels,
        limit_per_channel=limit_per_channel,
        days_back=days_back
    )
    
    # Export data if requested
    if export_csv:
        pipeline.export_data('csv')
    
    # Get final summary
    summary = pipeline.get_ingestion_summary()
    stats['summary'] = summary
    
    return stats 