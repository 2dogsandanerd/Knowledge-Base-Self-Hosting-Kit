"""Async IMAP/SMTP email client for generic email providers."""

import asyncio
from aioimaplib import aioimaplib
import aiosmtplib
import email
import re
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import make_msgid
from email_reply_parser import EmailReplyParser
from typing import List, Dict, Optional, Any
from loguru import logger

from .base_client import AbstractEmailClient


class IMAPClient(AbstractEmailClient):
    """Async IMAP/SMTP client for handling standard email servers."""

    def __init__(self, session_state: Dict[str, Any], config: Dict[str, Any]):
        self.config = config
        self.imap_client = None
        self.user = config.get('EMAIL_USER')
        self.password = config.get('EMAIL_PASSWORD')
        self.imap_host = config.get('IMAP_HOST')
        self.smtp_host = config.get('SMTP_HOST')
        self.imap_port = int(config.get('IMAP_PORT', 993))
        self.smtp_port = int(config.get('SMTP_PORT', 587))

        # Connection lock for thread safety
        self._connection_lock = asyncio.Lock()
        self._authenticated = False

    async def is_authenticated(self) -> bool:
        """Async IMAP authentication check and connect if needed"""
        if self._authenticated and self.imap_client:
            # Test connection
            try:
                await self.imap_client.noop()
                return True
            except Exception:
                logger.warning("IMAP connection lost, reconnecting...")
                self._authenticated = False

        # Connect and authenticate
        async with self._connection_lock:
            try:
                logger.info(f"Connecting to IMAP: {self.imap_host}:{self.imap_port}")

                # Create async IMAP client
                self.imap_client = aioimaplib.IMAP4_SSL(
                    host=self.imap_host,
                    port=self.imap_port
                )

                # Wait for server greeting
                await self.imap_client.wait_hello_from_server()

                # Login
                response = await self.imap_client.login(self.user, self.password)

                if response.result != 'OK':
                    logger.error(f"IMAP login failed: {response}")
                    return False

                logger.info("IMAP login successful")
                self._authenticated = True
                return True

            except Exception as e:
                logger.error(f"IMAP authentication failed: {e}")
                self._authenticated = False
                self.imap_client = None
                return False

    async def _ensure_connected(self) -> bool:
        """Ensure IMAP connection is alive"""
        if not self._authenticated:
            return await self.is_authenticated()

        try:
            await self.imap_client.noop()
            return True
        except Exception as e:
            logger.warning(f"IMAP connection check failed ({e}), reconnecting...")
            self._authenticated = False
            return await self.is_authenticated()

    def get_auth_url(self) -> Optional[str]:
        """Not applicable for IMAP"""
        return None

    def handle_auth_callback(self, code: str) -> bool:
        """Not applicable for IMAP"""
        return False

    async def get_unread_emails(self, max_results: int = 10, folder_name: str = "INBOX") -> List[Dict]:
        """Async fetch unread emails"""
        if not await self._ensure_connected():
            return []

        try:
            # Select mailbox
            response = await self.imap_client.select(f'"{folder_name}"')
            if response.result != 'OK':
                logger.error(f"Failed to select folder {folder_name}: {response}")
                return []

            logger.info(f"Folder {folder_name} selected")

            # Search for unread emails
            response = await self.imap_client.search('UNSEEN')

            if response.result != 'OK':
                logger.error("Failed to search for unread emails")
                return []

            logger.info(f"IMAP search response: {response}")

            # Parse email IDs
            email_ids_bytes = response.lines[0]
            if not email_ids_bytes:
                logger.info("No unread emails found")
                return []

            email_ids = email_ids_bytes.decode().split()

            # Fetch latest emails first (reversed)
            emails = []
            for email_id in reversed(email_ids[-max_results:]):
                details = await self.get_email_details(email_id)
                if details:
                    emails.append(details)

            logger.info(f"Fetched {len(emails)} unread emails")
            return emails

        except Exception as e:
            logger.error(f"Error fetching unread emails: {e}")
            return []

    async def get_emails(self, max_count: int = 20, folder_name: str = "INBOX") -> List[Dict[str, Any]]:
        """Async fetch emails from folder"""
        if not await self._ensure_connected():
            return []

        try:
            # Select mailbox
            response = await self.imap_client.select(f'"{folder_name}"')
            if response.result != 'OK':
                logger.error(f"Failed to select folder {folder_name}")
                return []

            # Search all emails
            response = await self.imap_client.search('ALL')

            if response.result != 'OK':
                logger.error("Failed to search emails")
                return []

            email_ids_bytes = response.lines[0]
            if not email_ids_bytes:
                return []

            email_ids = email_ids_bytes.decode().split()

            # Fetch latest
            emails = []
            for email_id in reversed(email_ids[-max_count:]):
                details = await self.get_email_details(email_id)
                if details:
                    emails.append(details)

            return emails

        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return []

    async def get_email_details(self, email_id: str) -> Optional[Dict]:
        """Async fetch single email details"""
        try:
            # Fetch email
            response = await self.imap_client.fetch(email_id, '(RFC822)')

            if response.result != 'OK':
                logger.error(f"Failed to fetch email {email_id}")
                return None

            # Parse response
            raw_email = response.lines[1]

            # Handle bytes, bytearray, or string
            if isinstance(raw_email, (bytes, bytearray)):
                # Convert bytearray to bytes if needed
                if isinstance(raw_email, bytearray):
                    raw_email = bytes(raw_email)
                msg = email.message_from_bytes(raw_email)
            elif isinstance(raw_email, str):
                msg = email.message_from_string(raw_email)
            else:
                logger.error(f"Unexpected email format: {type(raw_email)}")
                return None

            # Extract headers
            subject = self._decode_header(msg.get('Subject', ''))
            from_addr = self._decode_header(msg.get('From', ''))
            to_addr = self._decode_header(msg.get('To', ''))
            date = msg.get('Date', '')
            message_id = msg.get('Message-ID', '')
            in_reply_to = msg.get('In-Reply-To', '')
            references = msg.get('References', '')

            # Extract body
            body = self._get_email_body(msg)

            return {
                'id': email_id,
                'message_id': message_id,
                'thread_id': in_reply_to or message_id,
                'subject': subject,
                'from': from_addr,
                'to': to_addr,
                'date': date,
                'body': body,
                'snippet': body[:200] if body else '',
                'in_reply_to': in_reply_to,
                'references': references
            }

        except Exception as e:
            logger.error(f"Error fetching email {email_id}: {e}")
            return None

    async def get_thread_history(self, thread_id: str, max_messages: int = 10) -> List[Dict]:
        """Async get email thread (simplified - search by References)"""
        if not await self._ensure_connected():
            return []

        try:
            response = await self.imap_client.select('"INBOX"')
            if response.result != 'OK':
                return []

            # Search by Message-ID (simplified)
            response = await self.imap_client.search(f'HEADER Message-ID "{thread_id}"')

            if response.result != 'OK':
                return []

            email_ids_bytes = response.lines[0]
            if not email_ids_bytes:
                return []

            email_ids = email_ids_bytes.decode().split()

            thread = []
            for email_id in email_ids[:max_messages]:
                details = await self.get_email_details(email_id)
                if details:
                    thread.append(details)

            return thread

        except Exception as e:
            logger.error(f"Error fetching thread: {e}")
            return []

    async def create_draft(self, to: str, subject: str, body: str, thread_id: Optional[str] = None, in_reply_to: Optional[str] = None) -> Optional[str]:
        """Async create draft (save to Drafts folder)"""
        try:
            # Create MIME message
            msg = MIMEMultipart()
            msg['To'] = to
            msg['From'] = self.user
            msg['Subject'] = subject

            # Optional fields
            if in_reply_to:
                msg['In-Reply-To'] = in_reply_to
            if thread_id:
                msg['References'] = thread_id

            msg.attach(MIMEText(body, 'plain'))

            # Ensure connected
            if not await self._ensure_connected():
                raise Exception("Not authenticated")

            # Find drafts folder
            draft_folder = await self._find_draft_folder()
            if not draft_folder:
                draft_folder = 'Drafts'  # Default fallback

            # Append to Drafts folder
            response = await self.imap_client.append(
                f'"{draft_folder}"',
                msg.as_bytes()
            )

            if response.result != 'OK':
                raise Exception(f"Failed to create draft: {response}")

            logger.info("Draft created successfully")

            # Return draft ID (simplified - use subject hash as ID)
            return f"draft_{hash(subject)}"

        except Exception as e:
            logger.error(f"Error creating draft: {e}")
            return None

    async def move_to_label(self, message_id: str, label: str) -> bool:
        """Async move email to folder/label"""
        try:
            if not await self._ensure_connected():
                return False

            # Select INBOX first
            response = await self.imap_client.select('"INBOX"')
            if response.result != 'OK':
                return False

            # Copy to target folder
            response = await self.imap_client.copy(message_id, f'"{label}"')
            if response.result != 'OK':
                logger.error(f"Failed to copy to {label}")
                return False

            # Mark original as deleted
            response = await self.imap_client.store(message_id, '+FLAGS', '\\Deleted')
            if response.result != 'OK':
                logger.error("Failed to mark original as deleted")
                return False

            # Expunge (permanently delete)
            await self.imap_client.expunge()

            logger.info(f"Email {message_id} moved to {label}")
            return True

        except Exception as e:
            logger.error(f"Error moving email: {e}")
            return False

    async def check_draft_status(self, draft_id: str) -> str:
        """Async check draft status"""
        try:
            if not await self._ensure_connected():
                return "unknown"

            # Search in Drafts folder
            draft_folder = await self._find_draft_folder()
            if not draft_folder:
                return "unknown"

            response = await self.imap_client.select(f'"{draft_folder}"')
            if response.result != 'OK':
                return "unknown"

            # Search all drafts (simplified check)
            response = await self.imap_client.search('ALL')
            if response.result == 'OK' and response.lines[0]:
                return "exists"

            return "deleted"

        except Exception as e:
            logger.error(f"Error checking draft status: {e}")
            return "unknown"

    async def list_folders(self) -> List[str]:
        """Async list available folders"""
        try:
            if not await self._ensure_connected():
                return []

            response = await self.imap_client.list('""', '*')

            if response.result != 'OK':
                logger.error("Failed to list folders")
                return []

            folders = []
            for line in response.lines:
                if isinstance(line, bytes):
                    folder_name = self._decode_folder_name(line)
                    if folder_name:
                        folders.append(folder_name)

            logger.info(f"Found {len(folders)} folders")
            return folders

        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []

    async def get_user_email(self) -> Optional[str]:
        """Get user email address"""
        return self.user

    async def clear_inbox(self) -> Dict[str, Any]:
        """Async mark all emails in inbox as read"""
        try:
            if not await self._ensure_connected():
                return {"status": "error", "message": "Not connected"}

            response = await self.imap_client.select('"INBOX"')
            if response.result != 'OK':
                return {"status": "error", "message": "Failed to select INBOX"}

            # Search all emails
            response = await self.imap_client.search('ALL')
            if response.result != 'OK':
                return {"status": "error", "message": "Failed to search emails"}

            email_ids_bytes = response.lines[0]
            if not email_ids_bytes:
                return {"status": "success", "count": 0}

            email_ids = email_ids_bytes.decode().split()

            # Mark all as read
            for email_id in email_ids:
                await self.imap_client.store(email_id, '+FLAGS', '\\Seen')

            logger.info(f"Marked {len(email_ids)} emails as read")
            return {"status": "success", "count": len(email_ids)}

        except Exception as e:
            logger.error(f"Error clearing inbox: {e}")
            return {"status": "error", "message": str(e)}

    async def remove_label_from_email(self, message_id: str, folder_name: str) -> bool:
        """Async remove label/move email back to INBOX"""
        try:
            if not await self._ensure_connected():
                return False

            # Select folder
            response = await self.imap_client.select(f'"{folder_name}"')
            if response.result != 'OK':
                return False

            # Copy back to INBOX
            response = await self.imap_client.copy(message_id, '"INBOX"')
            if response.result != 'OK':
                logger.error("Failed to copy to INBOX")
                return False

            # Mark original as deleted
            response = await self.imap_client.store(message_id, '+FLAGS', '\\Deleted')
            if response.result != 'OK':
                logger.error("Failed to mark as deleted")
                return False

            # Expunge
            await self.imap_client.expunge()

            logger.info(f"Email {message_id} moved from {folder_name} to INBOX")
            return True

        except Exception as e:
            logger.error(f"Error removing label: {e}")
            return False

    async def get_emails_from_folder(self, folder_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Async get emails from specific folder"""
        return await self.get_emails(max_count=max_results, folder_name=folder_name)

    # Helper methods

    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""

        decoded_parts = decode_header(header)
        decoded_str = ""

        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                decoded_str += part.decode(encoding or 'utf-8', errors='ignore')
            else:
                decoded_str += part

        return decoded_str

    def _get_email_body(self, msg) -> str:
        """Extract email body from message"""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body = payload.decode(charset, errors='ignore')
                            break
                    except Exception as e:
                        logger.error(f"Error decoding part: {e}")
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='ignore')
            except Exception as e:
                logger.error(f"Error decoding body: {e}")

        return body.strip()

    def _decode_folder_name(self, folder_data: bytes) -> str:
        """Decode folder name from IMAP LIST response"""
        try:
            # Parse: b'(\\HasNoChildren) "/" "INBOX"'
            match = re.search(rb'"([^"]+)"$', folder_data)
            if match:
                return match.group(1).decode('utf-8', errors='ignore')
            return ""
        except Exception as e:
            logger.error(f"Error decoding folder name: {e}")
            return ""

    async def _find_draft_folder(self) -> Optional[str]:
        """Find the drafts folder name (can be 'Drafts', 'Draft', '[Gmail]/Drafts', etc.)"""
        try:
            folders = await self.list_folders()

            # Common draft folder names
            draft_names = ['Drafts', 'Draft', '[Gmail]/Drafts', 'INBOX.Drafts']

            for folder in folders:
                if folder in draft_names or 'draft' in folder.lower():
                    return folder

            return 'Drafts'  # Default fallback

        except Exception as e:
            logger.error(f"Error finding draft folder: {e}")
            return 'Drafts'

    async def close(self):
        """Close IMAP connection"""
        if self.imap_client:
            try:
                await self.imap_client.logout()
                logger.info("IMAP connection closed")
            except Exception as e:
                logger.error(f"Error closing IMAP: {e}")
            finally:
                self.imap_client = None
                self._authenticated = False
