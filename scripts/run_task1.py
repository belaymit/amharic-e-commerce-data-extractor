#!/usr/bin/env python3
"""Main script to run Task 1: Data Ingestion and Preprocessing."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import run_task_1
from src.config import app_config


async def main():
    """Main function to run Task 1."""
    print("=" * 60)
    print("🇪🇹 Amharic E-commerce Data Extractor - Task 1")
    print("=" * 60)
    print("Starting data ingestion and preprocessing...")
    print()
    
    # Configuration
    channels = app_config.target_channels
    limit_per_channel = 500  # Reduced for initial testing
    days_back = 7  # Start with 1 week of data
    
    print(f"📡 Target channels: {', '.join(channels)}")
    print(f"📊 Limit per channel: {limit_per_channel}")
    print(f"📅 Days back: {days_back}")
    print()
    
    try:
        # Run the data ingestion pipeline
        results = await run_task_1(
            channels=channels,
            limit_per_channel=limit_per_channel,
            days_back=days_back,
            export_csv=True
        )
        
        # Display results
        print("✅ Task 1 completed successfully!")
        print()
        print("📈 Results Summary:")
        print(f"  • Total messages scraped: {results['total_messages_scraped']}")
        print(f"  • Total messages processed: {results['total_messages_processed']}")
        print(f"  • Channels processed: {results['channels_processed']}")
        
        if results.get('errors'):
            print(f"  • Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        print()
        print("📊 Channel Breakdown:")
        for channel, stats in results['channel_stats'].items():
            print(f"  • {channel}: {stats['scraped']} scraped, {stats['processed']} processed")
        
        if results.get('summary'):
            summary = results['summary']
            print()
            print("💾 Data Storage:")
            print(f"  • Database: {summary['database_path']}")
            print(f"  • Total messages in DB: {summary['total_messages']}")
            print(f"  • CSV exports: data/processed/*.csv")
        
    except KeyboardInterrupt:
        print("\n⚠️  Task interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Task failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we're on Windows and need to set the event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 