import json
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class ChromeHistoryReader(BaseReader):
    """
    Chrome/Brave browser history reader that extracts browsing data from SQLite database.
    Supports reading from a copy to avoid locking issues.
    """

    def __init__(self) -> None:
        pass

    def load_data(
        self, chrome_profile_path: str | None = None, max_count: int = 1000
    ) -> list[Document]:
        docs: list[Document] = []

        if chrome_profile_path is None:
            # Default fallback for macOS
            chrome_profile_path = os.path.expanduser(
                "~/Library/Application Support/Google/Chrome/Default"
            )

        history_db_path = os.path.join(chrome_profile_path, "History")

        if not os.path.exists(history_db_path):
            print(f"⚠️ Browser history database not found at: {history_db_path}")
            return docs

        source_conn = None
        conn = None
        try:
            # SQLite's backup API captures committed rows from the main database and its WAL
            # while keeping the live browser database untouched.
            source_uri = f"{Path(history_db_path).resolve().as_uri()}?mode=ro"
            source_conn = sqlite3.connect(source_uri, uri=True)
            conn = sqlite3.connect(":memory:")
            source_conn.backup(conn)
            cursor = conn.cursor()

            query = """
            SELECT
                datetime(last_visit_time/1000000-11644473600,'unixepoch','localtime') as last_visit,
                url,
                title,
                visit_count,
                typed_count,
                hidden
            FROM urls
            ORDER BY last_visit_time DESC
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                if 0 < max_count <= len(docs):
                    break

                last_visit, url, title, visit_count, _typed_count, _hidden = row
                if not title or not url:
                    continue

                doc_content = f"""
[Title]: {title}
[URL]: {url}
[Last Visited]: {last_visit}
[Visits]: {visit_count}
"""
                doc = Document(text=doc_content, metadata={"title": title[0:150], "url": url})
                docs.append(doc)

        except Exception as e:
            print(f"❌ Error reading browser history: {e}")
            return docs
        finally:
            if conn is not None:
                conn.close()
            if source_conn is not None:
                source_conn.close()

        return docs

    @staticmethod
    def find_browser_paths() -> dict[str, Path]:
        """Find common browser profile base paths."""
        paths = {}
        home = Path.home()

        if os.name == "posix":  # macOS/Linux
            # macOS paths
            chrome = home / "Library/Application Support/Google/Chrome"
            brave = home / "Library/Application Support/BraveSoftware/Brave-Browser"
            if chrome.exists():
                paths["chrome"] = chrome
            if brave.exists():
                paths["brave"] = brave

        return paths


class IMessageReader(BaseReader):
    """
    iMessage data reader.

    Reads iMessage conversation data from the macOS Messages database (chat.db).
    Processes conversations into structured documents with metadata.
    """

    def __init__(self, concatenate_conversations: bool = True) -> None:
        """
        Initialize.

        Args:
            concatenate_conversations: Whether to concatenate messages within conversations for better context
        """
        self.concatenate_conversations = concatenate_conversations

    def _get_default_chat_db_path(self) -> Path:
        """
        Get the default path to the iMessage chat database.

        Returns:
            Path to the chat.db file
        """
        home = Path.home()
        return home / "Library" / "Messages" / "chat.db"

    def _convert_cocoa_timestamp(self, cocoa_timestamp: int) -> str:
        """
        Convert Cocoa timestamp to readable format.

        Args:
            cocoa_timestamp: Timestamp in Cocoa format (nanoseconds since 2001-01-01)

        Returns:
            Formatted timestamp string
        """
        if cocoa_timestamp == 0:
            return "Unknown"

        try:
            # Cocoa timestamp is nanoseconds since 2001-01-01 00:00:00 UTC
            # Convert to seconds and add to Unix epoch
            cocoa_epoch = datetime(2001, 1, 1)
            unix_timestamp = cocoa_timestamp / 1_000_000_000  # Convert nanoseconds to seconds
            message_time = cocoa_epoch.timestamp() + unix_timestamp
            return datetime.fromtimestamp(message_time).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return "Unknown"

    def _get_contact_name(self, handle_id: str) -> str:
        """
        Get a readable contact name from handle ID.

        Args:
            handle_id: The handle ID (phone number or email)

        Returns:
            Formatted contact name
        """
        if not handle_id:
            return "Unknown"

        # Clean up phone numbers and emails for display
        if "@" in handle_id:
            return handle_id  # Email address
        elif handle_id.startswith("+"):
            return handle_id  # International phone number
        else:
            # Try to format as phone number
            digits = "".join(filter(str.isdigit, handle_id))
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == "1":
                return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
            else:
                return handle_id

    def _read_messages_from_db(self, db_path: Path) -> list[dict]:
        """
        Read messages from the iMessage database.

        Args:
            db_path: Path to the chat.db file

        Returns:
            List of message dictionaries
        """
        if not db_path.exists():
            print(f"iMessage database not found at: {db_path}")
            return []

        try:
            # Connect to the database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Query to get messages with chat and handle information
            query = """
            SELECT
                m.ROWID as message_id,
                m.text,
                m.date,
                m.is_from_me,
                m.service,
                c.chat_identifier,
                c.display_name as chat_display_name,
                h.id as handle_id,
                c.ROWID as chat_id
            FROM message m
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY c.ROWID, m.date
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            messages = []
            for row in rows:
                (
                    message_id,
                    text,
                    date,
                    is_from_me,
                    service,
                    chat_identifier,
                    chat_display_name,
                    handle_id,
                    chat_id,
                ) = row

                message = {
                    "message_id": message_id,
                    "text": text,
                    "timestamp": self._convert_cocoa_timestamp(date),
                    "is_from_me": bool(is_from_me),
                    "service": service or "iMessage",
                    "chat_identifier": chat_identifier or "Unknown",
                    "chat_display_name": chat_display_name or "Unknown Chat",
                    "handle_id": handle_id or "Unknown",
                    "contact_name": self._get_contact_name(handle_id or ""),
                    "chat_id": chat_id,
                }
                messages.append(message)

            conn.close()
            return messages

        except sqlite3.Error as e:
            print(f"Error reading iMessage database: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error reading iMessage database: {e}")
            return []

    def _group_messages_by_chat(self, messages: list[dict]) -> dict[int, list[dict]]:
        """
        Group messages by chat ID.

        Args:
            messages: List of message dictionaries

        Returns:
            Dictionary mapping chat_id to list of messages
        """
        chats = {}
        for message in messages:
            chat_id = message["chat_id"]
            if chat_id not in chats:
                chats[chat_id] = []
            chats[chat_id].append(message)

        return chats

    def _create_concatenated_content(self, chat_id: int, messages: list[dict]) -> str:
        """
        Create concatenated content from chat messages.

        Args:
            chat_id: The chat ID
            messages: List of messages in the chat

        Returns:
            Concatenated text content
        """
        if not messages:
            return ""

        # Get chat info from first message
        first_msg = messages[0]
        chat_name = first_msg["chat_display_name"]
        chat_identifier = first_msg["chat_identifier"]

        # Build message content
        message_parts = []
        for message in messages:
            timestamp = message["timestamp"]
            is_from_me = message["is_from_me"]
            text = message["text"]
            contact_name = message["contact_name"]

            if is_from_me:
                prefix = "[You]"
            else:
                prefix = f"[{contact_name}]"

            if timestamp != "Unknown":
                prefix += f" ({timestamp})"

            message_parts.append(f"{prefix}: {text}")

        concatenated_text = "\n\n".join(message_parts)

        doc_content = f"""Chat: {chat_name}
Identifier: {chat_identifier}
Messages ({len(messages)} messages):

{concatenated_text}
"""
        return doc_content

    def _create_individual_content(self, message: dict) -> str:
        """
        Create content for individual message.

        Args:
            message: Message dictionary

        Returns:
            Formatted message content
        """
        timestamp = message["timestamp"]
        is_from_me = message["is_from_me"]
        text = message["text"]
        contact_name = message["contact_name"]
        chat_name = message["chat_display_name"]

        sender = "You" if is_from_me else contact_name

        return f"""Message from {sender} in chat "{chat_name}"
Time: {timestamp}
Content: {text}
"""

    def load_data(self, input_dir: str | None = None, **load_kwargs: Any) -> list[Document]:
        """
        Load iMessage data and return as documents.

        Args:
            input_dir: Optional path to directory containing chat.db file.
                      If not provided, uses default macOS location.
            **load_kwargs: Additional arguments (unused)

        Returns:
            List of Document objects containing iMessage data
        """
        docs = []

        # Determine database path
        if input_dir:
            db_path = Path(input_dir) / "chat.db"
        else:
            db_path = self._get_default_chat_db_path()

        # Read messages from database
        messages = self._read_messages_from_db(db_path)
        if not messages:
            return docs

        if self.concatenate_conversations:
            # Group messages by chat and create concatenated documents
            chats = self._group_messages_by_chat(messages)

            for chat_id, chat_messages in chats.items():
                if not chat_messages:
                    continue

                content = self._create_concatenated_content(chat_id, chat_messages)

                # Create metadata
                first_msg = chat_messages[0]
                last_msg = chat_messages[-1]

                metadata = {
                    "source": "iMessage",
                    "chat_id": chat_id,
                    "chat_name": first_msg["chat_display_name"],
                    "chat_identifier": first_msg["chat_identifier"],
                    "message_count": len(chat_messages),
                    "first_message_date": first_msg["timestamp"],
                    "last_message_date": last_msg["timestamp"],
                    "participants": list(
                        {msg["contact_name"] for msg in chat_messages if not msg["is_from_me"]}
                    ),
                }

                doc = Document(text=content, metadata=metadata)
                docs.append(doc)

        else:
            # Create individual documents for each message
            for message in messages:
                content = self._create_individual_content(message)

                metadata = {
                    "source": "iMessage",
                    "message_id": message["message_id"],
                    "chat_id": message["chat_id"],
                    "chat_name": message["chat_display_name"],
                    "chat_identifier": message["chat_identifier"],
                    "timestamp": message["timestamp"],
                    "is_from_me": message["is_from_me"],
                    "contact_name": message["contact_name"],
                    "service": message["service"],
                }

                doc = Document(text=content, metadata=metadata)
                docs.append(doc)

        return docs


class AppleMailReader(BaseReader):
    """Reader for Apple Mail data (macOS)."""

    def load_data(self, max_count: int = 1000) -> list[Document]:
        docs: list[Document] = []
        home = Path.home()
        mail_data_path = home / "Library/Mail/V10/MailData"
        envelope_index = mail_data_path / "Envelope Index"
        temp_db_path = "/tmp/leann_mail_index_copy"

        if not envelope_index.exists():
            # Try V9 if V10 doesn't exist
            mail_data_path = home / "Library/Mail/V9/MailData"
            envelope_index = mail_data_path / "Envelope Index"
            if not envelope_index.exists():
                print("⚠️ Apple Mail Envelope Index not found.")
                return docs

        try:
            shutil.copy2(envelope_index, temp_db_path)
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Query to get message subjects, senders, and content previews
            query = """
            SELECT
                m.subject,
                m.sender,
                datetime(m.date_sent, 'unixepoch', '31 years') as date,
                s.snippet
            FROM messages m
            LEFT JOIN message_snippets s ON m.ROWID = s.message_id
            ORDER BY m.date_sent DESC
            LIMIT ?
            """
            cursor.execute(query, (max_count,))
            rows = cursor.fetchall()

            for row in rows:
                subject, sender, date, snippet = row
                if not subject and not snippet:
                    continue

                content = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{snippet or ''}"
                docs.append(
                    Document(
                        text=content, metadata={"subject": subject or "", "sender": sender or ""}
                    )
                )

            conn.close()
            os.remove(temp_db_path)
        except Exception as e:
            print(f"❌ Error reading Apple Mail: {e}")
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

        return docs


class AppleCalendarReader(BaseReader):
    """Reader for Apple Calendar events (macOS)."""

    def load_data(self, max_count: int = 1000) -> list[Document]:
        docs: list[Document] = []
        home = Path.home()
        calendar_cache = home / "Library/Calendars/Calendar Cache"
        temp_db_path = "/tmp/leann_calendar_index_copy"

        if not calendar_cache.exists():
            print("⚠️ Apple Calendar Cache not found.")
            return docs

        try:
            shutil.copy2(calendar_cache, temp_db_path)
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Query for events
            query = """
            SELECT
                summary,
                description,
                location,
                datetime(start_date + 978307200, 'unixepoch', 'localtime') as start,
                datetime(end_date + 978307200, 'unixepoch', 'localtime') as end
            FROM CI_EVENT
            ORDER BY start_date DESC
            LIMIT ?
            """
            cursor.execute(query, (max_count,))
            rows = cursor.fetchall()

            for row in rows:
                summary, description, location, start, end = row
                if not summary:
                    continue

                content = f"Event: {summary}\nStart: {start}\nEnd: {end}\nLocation: {location or ''}\nDescription: {description or ''}"
                docs.append(Document(text=content, metadata={"event": summary, "start": start}))

            conn.close()
            os.remove(temp_db_path)
        except Exception as e:
            print(f"❌ Error reading Apple Calendar: {e}")
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

        return docs


class WeChatHistoryReader(BaseReader):
    """
    WeChat chat history reader that extracts chat data from exported JSON files.
    """

    def __init__(self) -> None:
        """Initialize."""
        pass

    def _extract_readable_text(self, content: Any) -> str:
        if not content:
            return ""
        if isinstance(content, dict):
            text_parts = [
                str(content.get(f, ""))
                for f in ["title", "quoted", "content", "text"]
                if content.get(f)
            ]
            return " | ".join(text_parts) if text_parts else ""
        if not isinstance(content, str):
            return ""
        clean_content = re.sub(r"^wxid_[^:]+:\s*", "", content)
        clean_content = re.sub(r"^[^:]+:\s*", "", clean_content)
        if clean_content.strip().startswith("<") or "recalled a message" in clean_content:
            return ""
        return clean_content.strip()

    def _is_text_message(self, content: Any) -> bool:
        if not content:
            return False
        if isinstance(content, dict):
            return any(content.get(f) for f in ["title", "quoted", "content", "text"])
        if not isinstance(content, str):
            return False
        if any(tag in content for tag in ["<img", "<emoji", "<voice", "<video", "<appmsg"]):
            return False
        if "recalled a message" in content:
            return False
        clean_content = re.sub(r"^wxid_[^:]+:\s*", "", content)
        clean_content = re.sub(r"^[^:]+:\s*", "", clean_content)
        return len(clean_content.strip()) > 0 and not clean_content.strip().startswith("<")

    def load_data(
        self, wechat_export_dir: str, max_count: int = 1000, concatenate_messages: bool = True
    ) -> list[Document]:
        docs: list[Document] = []
        if not os.path.exists(wechat_export_dir):
            print(f"WeChat export directory not found at: {wechat_export_dir}")
            return docs

        try:
            json_files = list(Path(wechat_export_dir).glob("*.json"))
            count = 0
            for json_file in json_files:
                if 0 < max_count <= count:
                    break
                try:
                    with open(json_file, encoding="utf-8") as f:
                        chat_data = json.load(f)
                    contact_name = json_file.stem

                    messages_text = []
                    for message in chat_data:
                        content = message.get("content", "")
                        if self._is_text_message(content):
                            readable_text = self._extract_readable_text(content) or message.get(
                                "message", ""
                            )
                            if readable_text.strip():
                                create_time = message.get("createTime", 0)
                                time_str = (
                                    datetime.fromtimestamp(create_time).strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                    if create_time
                                    else "Unknown"
                                )
                                sender = "[Me]" if message.get("isSentFromSelf") else "[Contact]"
                                messages_text.append(f"({time_str}) {sender}: {readable_text}")

                    if messages_text:
                        if concatenate_messages:
                            full_text = f"Contact: {contact_name}\n\n" + "\n".join(messages_text)
                            docs.append(
                                Document(text=full_text, metadata={"contact_name": contact_name})
                            )
                            count += 1
                        else:
                            for msg in messages_text:
                                if 0 < max_count <= count:
                                    break
                                docs.append(
                                    Document(text=msg, metadata={"contact_name": contact_name})
                                )
                                count += 1
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
        except Exception as e:
            print(f"Error reading WeChat history: {e}")

        return docs


class SlackMCPReader:
    """Reader for Slack data via MCP servers."""

    def __init__(
        self,
        mcp_server_command: str,
        workspace_name: str | None = None,
        concatenate_conversations: bool = True,
    ):
        self.mcp_server_command = mcp_server_command
        self.workspace_name = workspace_name
        self.concatenate_conversations = concatenate_conversations

    async def load_data(self, channels: list[str] | None = None) -> list[Document]:
        return []


class TwitterMCPReader:
    """Reader for Twitter bookmarks via MCP servers."""

    def __init__(self, mcp_server_command: str):
        self.mcp_server_command = mcp_server_command

    async def load_data(self, max_bookmarks: int = 100) -> list[Document]:
        return []


class ChatGPTReader(BaseReader):
    """Reader for ChatGPT export files (.html or .zip)."""

    def load_data(self, export_path: str) -> list[Document]:
        return []


class ClaudeReader(BaseReader):
    """Reader for Claude export files (.json or .zip)."""

    def load_data(self, export_path: str) -> list[Document]:
        return []
