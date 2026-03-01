"""Factory for creating email client instances.

Gmail OAuth removed - use IMAP for all providers including Gmail.
For Gmail: Use App Password from https://myaccount.google.com/apppasswords
"""

from typing import Dict, Any, Optional

from .base_client import AbstractEmailClient
# Gmail OAuth removed - use IMAP for all providers
from .imap_client import IMAPClient

def get_email_client(config: Dict[str, Any], session_state: Dict[str, Any]) -> Optional[AbstractEmailClient]:
    """
    Factory function to get IMAP email client.

    Gmail OAuth removed. Use IMAP for all providers (Gmail, Outlook, etc.)

    Args:
        config: The application configuration dictionary.
        session_state: The Streamlit session state dictionary (deprecated, kept for compatibility).

    Returns:
        An instance of IMAPClient, or None if config is missing.
    """
    provider = config.get("EMAIL_PROVIDER", "imap").lower()

    if provider == "imap":
        return IMAPClient(session_state=session_state, config=config)
    elif provider in ["google", "gmail"]:
        # Gmail OAuth deprecated - warn user
        print("⚠️  Gmail OAuth is deprecated. Please use IMAP instead.")
        print("Set EMAIL_PROVIDER=imap and use Gmail App Password.")
        print("See: https://myaccount.google.com/apppasswords")
        return None
    else:
        # Only IMAP supported
        print(f"⚠️  Unknown provider '{provider}'. Only 'imap' is supported.")
        return None
