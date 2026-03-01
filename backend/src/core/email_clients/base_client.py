"""Abstract Base Class for all email clients (async)."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class AbstractEmailClient(ABC):
    """Abstract base class defining async interface for all email clients."""

    @abstractmethod
    def __init__(self, session_state: Dict[str, Any], config: Dict[str, Any]):
        """Initialize client with configuration and session state."""
        pass

    @abstractmethod
    async def is_authenticated(self) -> bool:
        """Check if client is authenticated and connected (async)."""
        pass

    @abstractmethod
    def get_auth_url(self) -> Optional[str]:
        """Get authentication URL for OAuth flows (not async - just returns URL)."""
        pass

    @abstractmethod
    def handle_auth_callback(self, code: str) -> bool:
        """Handle OAuth callback (not async - just processes code)."""
        pass

    @abstractmethod
    async def get_unread_emails(self, max_results: int = 10, folder_name: str = "INBOX") -> List[Dict]:
        """Fetch unread emails from specific folder (async)."""
        pass

    @abstractmethod
    async def get_emails(self, max_count: int = 20, folder_name: str = "INBOX") -> List[Dict[str, Any]]:
        """Fetch list of emails from specific folder (async)."""
        pass

    @abstractmethod
    async def get_email_details(self, message_id: str) -> Optional[Dict]:
        """Fetch details of specific email (async)."""
        pass

    @abstractmethod
    async def create_draft(self, to: str, subject: str, body: str, thread_id: Optional[str] = None, in_reply_to: Optional[str] = None) -> Optional[str]:
        """Create email draft (async)."""
        pass

    @abstractmethod
    async def get_thread_history(self, thread_id: str, max_messages: int = 10) -> List[Dict]:
        """Fetch email thread history (async)."""
        pass

    @abstractmethod
    async def move_to_label(self, message_id: str, label: str) -> bool:
        """Move email to specific folder/label (async)."""
        pass

    @abstractmethod
    async def check_draft_status(self, draft_id: str) -> str:
        """Check draft status (e.g. 'exists', 'deleted') (async)."""
        pass

    @abstractmethod
    async def list_folders(self) -> List[str]:
        """Get list of available folders/labels (async)."""
        pass

    @abstractmethod
    async def get_user_email(self) -> Optional[str]:
        """Get authenticated user's email address (async).

        Returns:
            Email address as string or None if not available.
        """
        pass

    @abstractmethod
    async def clear_inbox(self) -> Dict[str, Any]:
        """Mark all emails in inbox as read or move them (async).

        Returns:
            Dictionary with result, e.g. {'status': 'success', 'count': 10}.
        """
        pass

    @abstractmethod
    async def remove_label_from_email(self, message_id: str, folder_or_label_name: str) -> bool:
        """Remove a label/folder from an email (async).

        For Gmail: Removes the label from the message
        For IMAP: Moves the message from the folder back to INBOX

        Args:
            message_id: Email message ID or UID
            folder_or_label_name: Label name (Gmail) or folder name (IMAP)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_emails_from_folder(
        self,
        folder_name: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Get emails from a specific folder/label (async)."""
        pass
