"""
Email Loaders for .eml and .mbox formats.

This module provides specialized loaders for email formats:
- EmailLoader: Single .eml files (RFC 822)
- MboxLoader: mbox archives (Gmail Takeout, Thunderbird)

Uses Python Standard Library (email, mailbox) - no external dependencies.
Compatible with LlamaIndex Document format.
"""

from pathlib import Path
from typing import List, Optional
import email
from email import policy
from email.parser import Parser, BytesParser
import mailbox
from llama_index.core.schema import Document
from loguru import logger


class EmailLoader:
    """
    Loader for single email files (.eml format).

    Supports:
    - RFC 822 email format
    - Multipart messages (text/plain extraction)
    - Header metadata extraction
    - UTF-8 encoding with fallback

    Example:
        >>> loader = EmailLoader("path/to/email.eml")
        >>> docs = loader.load()
        >>> print(docs[0].metadata["subject"])
    """

    def __init__(self, file_path: str):
        """
        Initialize EmailLoader.

        Args:
            file_path: Path to .eml file
        """
        self.file_path = Path(file_path)
        self.logger = logger.bind(component="EmailLoader")

    def load(self) -> List[Document]:
        """
        Load .eml file and extract metadata + body.

        Returns:
            List with single Document containing email body and metadata

        Raises:
            FileNotFoundError: If email file doesn't exist
            Exception: If email parsing fails
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Email file not found: {self.file_path}")

        try:
            # Try UTF-8 first, fallback to binary with policy
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    msg = Parser(policy=policy.default).parse(f)
            except UnicodeDecodeError:
                # Fallback to binary mode
                with open(self.file_path, 'rb') as f:
                    msg = BytesParser(policy=policy.default).parsebytes(f.read())

            # Extract metadata
            metadata = {
                'source': str(self.file_path),
                'file_type': '.eml',
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'cc': msg.get('Cc', ''),
                'bcc': msg.get('Bcc', ''),
                'subject': msg.get('Subject', ''),
                'date': msg.get('Date', ''),
                'message_id': msg.get('Message-ID', ''),
                'in_reply_to': msg.get('In-Reply-To', ''),  # Thread detection
                'references': msg.get('References', '')     # Thread detection
            }

            # Extract body (prefer plain text over HTML)
            body = self._extract_body(msg)

            if not body.strip():
                self.logger.warning(f"Empty email body: {self.file_path}")

            # Create LlamaIndex Document
            doc = Document(
                text=body,
                metadata=metadata
            )

            self.logger.info(
                f"Loaded email: '{metadata['subject']}' "
                f"from {metadata['from']} ({len(body)} chars)"
            )

            return [doc]

        except Exception as e:
            self.logger.error(f"Failed to load email {self.file_path}: {e}")
            raise

    def _extract_body(self, msg) -> str:
        """
        Extract plain text body from email message.

        Handles:
        - Multipart messages (walks all parts)
        - HTML-only emails (extracts HTML as fallback)
        - Encoding issues (uses 'ignore' error handling)

        Args:
            msg: email.message.Message object

        Returns:
            Extracted body text (empty string if no text found)
        """
        body_parts = []

        if msg.is_multipart():
            # Walk through all parts, prioritize text/plain
            for part in msg.walk():
                content_type = part.get_content_type()

                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(
                                payload.decode('utf-8', errors='ignore')
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to decode part: {e}")

            # Fallback to HTML if no plain text found
            if not body_parts:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                html_body = payload.decode('utf-8', errors='ignore')
                                # Simple HTML strip (could use html2text for better results)
                                body_parts.append(html_body)
                        except Exception as e:
                            self.logger.warning(f"Failed to decode HTML: {e}")
        else:
            # Single-part message
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(
                        payload.decode('utf-8', errors='ignore')
                    )
            except Exception as e:
                self.logger.warning(f"Failed to decode message: {e}")

        return "\n\n".join(body_parts)


class MboxLoader:
    """
    Loader for mbox archive files (Gmail Takeout, Thunderbird).

    Supports:
    - mbox format (RFC 4155)
    - Multiple emails per file
    - Configurable max emails (to prevent memory issues)
    - Same metadata extraction as EmailLoader

    Example:
        >>> loader = MboxLoader("path/to/archive.mbox", max_emails=1000)
        >>> docs = loader.load()
        >>> print(f"Loaded {len(docs)} emails")
    """

    def __init__(self, file_path: str, max_emails: Optional[int] = 10000):
        """
        Initialize MboxLoader.

        Args:
            file_path: Path to .mbox file
            max_emails: Maximum number of emails to load (default: 10000)
                       Set to None for unlimited (use with caution!)
        """
        self.file_path = Path(file_path)
        self.max_emails = max_emails
        self.logger = logger.bind(component="MboxLoader")

    def load(self) -> List[Document]:
        """
        Load mbox archive and return list of Documents.

        Returns:
            List of Documents (one per email)

        Raises:
            FileNotFoundError: If mbox file doesn't exist
            Exception: If mbox parsing fails
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Mbox file not found: {self.file_path}")

        documents = []

        try:
            mbox = mailbox.mbox(str(self.file_path))

            for idx, message in enumerate(mbox):
                # Check max_emails limit
                if self.max_emails is not None and idx >= self.max_emails:
                    self.logger.warning(
                        f"Reached max_emails limit: {self.max_emails}. "
                        f"Skipping remaining emails."
                    )
                    break

                try:
                    doc = self._process_message(message, idx)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to process email {idx} in mbox: {e}. Skipping."
                    )
                    continue

            self.logger.info(
                f"Loaded {len(documents)} emails from mbox: {self.file_path}"
            )

            return documents

        except Exception as e:
            self.logger.error(f"Failed to load mbox {self.file_path}: {e}")
            raise

    def _process_message(self, message, idx: int) -> Optional[Document]:
        """
        Process single email from mbox into Document.

        Args:
            message: mailbox.mboxMessage object
            idx: Email index in mbox

        Returns:
            Document or None if email has no body
        """
        # Extract metadata
        metadata = {
            'source': str(self.file_path),
            'file_type': '.mbox',
            'email_index': idx,
            'from': message.get('From', ''),
            'to': message.get('To', ''),
            'cc': message.get('Cc', ''),
            'subject': message.get('Subject', ''),
            'date': message.get('Date', ''),
            'message_id': message.get('Message-ID', ''),
            'in_reply_to': message.get('In-Reply-To', ''),
            'references': message.get('References', '')
        }

        # Extract body using same logic as EmailLoader
        body = self._extract_body(message)

        if not body.strip():
            # Skip empty emails
            return None

        return Document(
            text=body,
            metadata=metadata
        )

    def _extract_body(self, msg) -> str:
        """
        Extract plain text body from mbox message.

        Same logic as EmailLoader._extract_body().

        Args:
            msg: mailbox.mboxMessage object

        Returns:
            Extracted body text
        """
        body_parts = []

        if msg.is_multipart():
            # Prioritize text/plain
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(
                                payload.decode('utf-8', errors='ignore')
                            )
                    except Exception:
                        pass

            # Fallback to HTML
            if not body_parts:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                body_parts.append(
                                    payload.decode('utf-8', errors='ignore')
                                )
                        except Exception:
                            pass
        else:
            # Single-part
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(
                        payload.decode('utf-8', errors='ignore')
                    )
            except Exception:
                pass

        return "\n\n".join(body_parts)
