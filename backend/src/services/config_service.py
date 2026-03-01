
import os
from dotenv import dotenv_values, set_key, unset_key
from typing import Dict, Any
from loguru import logger

class ConfigService:
    """Manages loading and saving of .env file configurations."""

    def __init__(self, dotenv_path: str = ".env"):
        self.dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", dotenv_path)
        logger.debug(f"ConfigService initialized. Looking for .env at: {self.dotenv_path}")
        if not os.path.exists(self.dotenv_path):
            logger.warning(f".env file not found at {self.dotenv_path}. Creating an empty one.")
            open(self.dotenv_path, 'a').close()

    def load_configuration(self) -> Dict[str, Any]:
        """
        Loads the configuration from the .env file and merges with OS environment variables.
        OS environment variables take precedence.

        Returns:
            A dictionary containing the current configuration.
        """
        logger.info(f"Loading configuration from {self.dotenv_path}")
        config = dotenv_values(self.dotenv_path)
        
        # Merge with os.environ (OS env vars take precedence)
        for key, value in os.environ.items():
            config[key] = value
            
        logger.debug(f"Loaded configuration (merged with ENV): {config}")
        return config

    def save_configuration(self, config_data: Dict[str, Any]) -> bool:
        """
        Saves the given configuration data to the .env file.

        Args:
            config_data: The configuration dictionary to save.

        Returns:
            True if successful, False otherwise.
        """
        logger.info(f"Saving configuration to {self.dotenv_path}")
        try:
            # Migration: DATABASE_URL -> lea_database_url
            if 'DATABASE_URL' in config_data and not config_data.get('lea_database_url'):
                config_data['lea_database_url'] = config_data['DATABASE_URL']

            # Ensure old key is removed from .env
            if 'DATABASE_URL' in config_data:
                try:
                    unset_key(self.dotenv_path, 'DATABASE_URL')
                except Exception as e:
                    logger.warning(f"Unable to unset DATABASE_URL: {e}")

            for key, value in config_data.items():
                # set_key handles creating the file if it doesn't exist
                # and updates the key if it exists, or adds it if it doesn't.
                set_key(self.dotenv_path, key, str(value))
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

config_service = ConfigService()
