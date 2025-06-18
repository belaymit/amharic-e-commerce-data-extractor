"""Configuration settings for the Amharic E-commerce Data Extractor."""

from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class TelegramConfig(BaseSettings):
    """Telegram API configuration."""
    
    api_id: Optional[int] = Field(None, env="TELEGRAM_API_ID")
    api_hash: Optional[str] = Field(None, env="TELEGRAM_API_HASH")
    phone_number: Optional[str] = Field(None, env="TELEGRAM_PHONE_NUMBER")
    session_name: str = Field("ethiomart_scraper", env="TELEGRAM_SESSION_NAME")
    
    model_config = {"env_file": ".env", "extra": "ignore"}


class DataConfig(BaseSettings):
    """Data storage and processing configuration."""
    
    base_data_dir: Path = Field(Path("data"), env="DATA_DIR")
    raw_data_dir: Path = Field(Path("data/raw"), env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(Path("data/processed"), env="PROCESSED_DATA_DIR")
    database_path: Path = Field(Path("data/ethiomart.db"), env="DATABASE_PATH")
    
    # Processing settings
    batch_size: int = Field(100, env="BATCH_SIZE")
    max_messages: int = Field(10000, env="MAX_MESSAGES")
    
    model_config = {"env_file": ".env", "extra": "ignore"}


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # Default Ethiopian e-commerce Telegram channels
    target_channels: List[str] = Field(
        default=[
            "@ShegerOnlineStore",
            "@ethio_commerce",
            "@addis_market",
            "@ethiopia_shopping",
            "@bole_trading"
        ]
    )
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Path = Field(Path("logs/ethiomart.log"), env="LOG_FILE")
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    # Create subdirectories
    def __post_init__(self):
        """Create necessary directories."""
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Path) and attr_name.endswith('_dir'):
                attr_value.mkdir(parents=True, exist_ok=True)


# Global configuration instances
telegram_config = TelegramConfig()
data_config = DataConfig()
app_config = AppConfig()

# Create directories
data_config.base_data_dir.mkdir(parents=True, exist_ok=True)
data_config.raw_data_dir.mkdir(parents=True, exist_ok=True)
data_config.processed_data_dir.mkdir(parents=True, exist_ok=True)
app_config.log_file.parent.mkdir(parents=True, exist_ok=True) 