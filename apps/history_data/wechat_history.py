import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class WeChatHistoryReader(BaseReader):
    """
    WeChat chat history reader that extracts chat data from exported JSON files.

    Reads WeChat chat history from exported JSON files (from wechat-exporter tool)
    and creates documents with embedded metadata similar to the Chrome history reader structure.

    Also includes utilities for automatic WeChat chat history export.
    """

    def __init__(self) -> None:
        """Initialize."""
        self.packages_dir = Path(__file__).parent.parent.parent / "packages"
        self.wechat_exporter_dir = self.packages_dir / "wechat-exporter"
        self.wechat_decipher_dir = self.packages_dir / "wechat-decipher-macos"

    def check_wechat_running(self) -> bool:
        """Check if WeChat is currently running."""
        try:
            result = subprocess.run(["pgrep", "-f", "WeChat"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def install_wechattweak(self) -> bool:
        """Install WeChatTweak CLI tool."""
        try:
            # Create wechat-exporter directory if it doesn't exist
            self.wechat_exporter_dir.mkdir(parents=True, exist_ok=True)

            wechattweak_path = self.wechat_exporter_dir / "wechattweak-cli"
            if not wechattweak_path.exists():
                print("Downloading WeChatTweak CLI...")
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        "-o",
                        str(wechattweak_path),
                        "https://github.com/JettChenT/WeChatTweak-CLI/releases/latest/download/wechattweak-cli",
                    ],
                    check=True,
                )

            # Make executable
            wechattweak_path.chmod(0o755)

            # Install WeChatTweak
            print("Installing WeChatTweak...")
            subprocess.run(["sudo", str(wechattweak_path), "install"], check=True)
            return True
        except Exception as e:
            print(f"Error installing WeChatTweak: {e}")
            return False

    def restart_wechat(self):
        """Restart WeChat to apply WeChatTweak."""
        try:
            print("Restarting WeChat...")
            subprocess.run(["pkill", "-f", "WeChat"], check=False)
            time.sleep(2)
            subprocess.run(["open", "-a", "WeChat"], check=True)
            time.sleep(5)  # Wait for WeChat to start
        except Exception as e:
            print(f"Error restarting WeChat: {e}")

    def check_api_available(self) -> bool:
        """Check if WeChatTweak API is available."""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:48065/wechat/allcontacts"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def _extract_readable_text(self, content: str) -> str:
        """
        Extract readable text from message content, removing XML and system messages.

        Args:
            content: The raw message content (can be string or dict)

        Returns:
            Cleaned, readable text
        """
        if not content:
            return ""

        # Handle dictionary content (like quoted messages)
        if isinstance(content, dict):
            # Extract text from dictionary structure
            text_parts = []
            if "title" in content:
                text_parts.append(str(content["title"]))
            if "quoted" in content:
                text_parts.append(str(content["quoted"]))
            if "content" in content:
                text_parts.append(str(content["content"]))
            if "text" in content:
                text_parts.append(str(content["text"]))

            if text_parts:
                return " | ".join(text_parts)
            else:
                # If we can't extract meaningful text from dict, return empty
                return ""

        # Handle string content
        if not isinstance(content, str):
            return ""

        # Remove common prefixes like "wxid_xxx:\n"
        clean_content = re.sub(r"^wxid_[^:]+:\s*", "", content)
        # Do not strip a generic "^word:" prefix here: the exporter writes message
        # bodies straight into `content`, so a leading colon prefix is ordinary text
        # ("Reminder: buy milk" -> "buy milk", "Note:" -> "").

        # If it's just XML or system message, return empty
        if clean_content.strip().startswith("<") or "recalled a message" in clean_content:
            return ""

        return clean_content.strip()

    def _is_text_message(self, content: str) -> bool:
        """
        Check if a message contains readable text content.

        Args:
            content: The message content (can be string or dict)

        Returns:
            True if the message contains readable text, False otherwise
        """
        if not content:
            return False

        # Handle dictionary content
        if isinstance(content, dict):
            # Check if dict has any readable text fields
            text_fields = ["title", "quoted", "content", "text"]
            for field in text_fields:
                if content.get(field):
                    return True
            return False

        # Handle string content
        if not isinstance(content, str):
            return False

        # Skip image messages (contain XML with img tags)
        if "<img" in content and "cdnurl" in content:
            return False

        # Skip emoji messages (contain emoji XML tags)
        if "<emoji" in content and "productid" in content:
            return False

        # Skip voice messages
        if "<voice" in content:
            return False

        # Skip video messages
        if "<video" in content:
            return False

        # Skip file messages
        if "<appmsg" in content and "appid" in content:
            return False

        # Skip system messages (like "recalled a message")
        if "recalled a message" in content:
            return False

        # Remove common prefixes like "wxid_xxx:\n" and check for actual content
        clean_content = re.sub(r"^wxid_[^:]+:\s*", "", content)
        # Same as _extract_readable_text: a generic "^word:" strip would empty
        # messages like "Note:" and drop them from the index.

        # If after cleaning we have meaningful text, consider it readable
        if len(clean_content.strip()) > 0 and not clean_content.strip().startswith("<"):
            return True

        return False

    def _concatenate_messages(
        self,
        messages: list[dict],
        max_length: int = 128,
        time_window_minutes: int = 30,
        overlap_messages: int = 0,
    ) -> list[dict]:
        """
        Concatenate messages based on length and time rules.

        Args:
            messages: List of message dictionaries
            max_length: Maximum length for concatenated message groups. Use -1 to disable length constraint.
            time_window_minutes: Time window in minutes to group messages together. Use -1 to disable time constraint.
            overlap_messages: Number of messages to overlap between consecutive groups

        Returns:
            List of concatenated message groups
        """
        if not messages:
            return []

        concatenated_groups = []
        current_group = []
        current_length = 0
        last_timestamp = None

        for message in messages:
            # Extract message info
            content = message.get("content", "")
            message_text = message.get("message", "")
            create_time = message.get("createTime", 0)
            message.get("fromUser", "")
            message.get("toUser", "")
            message.get("isSentFromSelf", False)

            # Extract readable text
            readable_text = self._extract_readable_text(content)
            if not readable_text:
                readable_text = message_text

            # Skip empty messages
            if not readable_text.strip():
                continue

            # Check time window constraint (only if time_window_minutes != -1)
            if time_window_minutes != -1 and last_timestamp is not None and create_time > 0:
                time_diff_minutes = (create_time - last_timestamp) / 60
                if time_diff_minutes > time_window_minutes:
                    # Time gap too large, start new group
                    if current_group:
                        concatenated_groups.append(
                            {
                                "messages": current_group,
                                "total_length": current_length,
                                "start_time": current_group[0].get("createTime", 0),
                                "end_time": current_group[-1].get("createTime", 0),
                            }
                        )
                        # Keep last few messages for overlap
                        if overlap_messages > 0 and len(current_group) > overlap_messages:
                            current_group = current_group[-overlap_messages:]
                            current_length = sum(
                                len(
                                    self._extract_readable_text(msg.get("content", ""))
                                    or msg.get("message", "")
                                )
                                for msg in current_group
                            )
                        else:
                            current_group = []
                            current_length = 0

            # Check length constraint (only if max_length != -1)
            message_length = len(readable_text)
            if max_length != -1 and current_length + message_length > max_length and current_group:
                # Current group would exceed max length, save it and start new
                concatenated_groups.append(
                    {
                        "messages": current_group,
                        "total_length": current_length,
                        "start_time": current_group[0].get("createTime", 0),
                        "end_time": current_group[-1].get("createTime", 0),
                    }
                )
                # Keep last few messages for overlap
                if overlap_messages > 0 and len(current_group) > overlap_messages:
                    current_group = current_group[-overlap_messages:]
                    current_length = sum(
                        len(
                            self._extract_readable_text(msg.get("content", ""))
                            or msg.get("message", "")
                        )
                        for msg in current_group
                    )
                else:
                    current_group = []
                    current_length = 0

            # Add message to current group
            current_group.append(message)
            current_length += message_length
            last_timestamp = create_time

        # Add the last group if it exists
        if current_group:
            concatenated_groups.append(
                {
                    "messages": current_group,
                    "total_length": current_length,
                    "start_time": current_group[0].get("createTime", 0),
                    "end_time": current_group[-1].get("createTime", 0),
                }
            )

        return concatenated_groups

    def _create_concatenated_content(
        self, message_group: dict, contact_name: str
    ) -> tuple[str, str]:
        """
        Create concatenated content from a group of messages.

        Args:
            message_group: Dictionary containing messages and metadata
            contact_name: Name of the contact

        Returns:
            Formatted concatenated content
        """
        messages = message_group["messages"]
        start_time = message_group["start_time"]
        end_time = message_group["end_time"]

        # Format timestamps
        if start_time:
            try:
                start_timestamp = datetime.fromtimestamp(start_time)
                start_time_str = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError):
                start_time_str = str(start_time)
        else:
            start_time_str = "Unknown"

        if end_time:
            try:
                end_timestamp = datetime.fromtimestamp(end_time)
                end_time_str = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError):
                end_time_str = str(end_time)
        else:
            end_time_str = "Unknown"

        # Build concatenated message content
        message_parts = []
        for message in messages:
            content = message.get("content", "")
            message_text = message.get("message", "")
            create_time = message.get("createTime", 0)
            is_sent_from_self = message.get("isSentFromSelf", False)

            # Extract readable text
            readable_text = self._extract_readable_text(content)
            if not readable_text:
                readable_text = message_text

            # Format individual message
            if create_time:
                try:
                    timestamp = datetime.fromtimestamp(create_time)
                    # change to YYYY-MM-DD HH:MM:SS
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, OSError):
                    time_str = str(create_time)
            else:
                time_str = "Unknown"

            sender = "[Me]" if is_sent_from_self else "[Contact]"
            message_parts.append(f"({time_str}) {sender}: {readable_text}")

        concatenated_text = "\n".join(message_parts)

        # Create final document content
        doc_content = f"""
Contact: {contact_name}
Time Range: {start_time_str} - {end_time_str}
Messages ({len(messages)} messages, {message_group["total_length"]} chars):

{concatenated_text}
"""
        # TODO @yichuan give better format and rich info here!
        doc_content = f"""
{concatenated_text}
"""
        return doc_content, contact_name

    def load_data(self, input_dir: str | None = None, **load_kwargs: Any) -> list[Document]:
        """
        Load WeChat chat history data from exported JSON files.

        Args:
            input_dir: Directory containing exported WeChat JSON files
            **load_kwargs:
                max_count (int): Maximum amount of chat entries to read.
                wechat_export_dir (str): Custom path to WeChat export directory.
                include_non_text (bool): Whether to include non-text messages (images, emojis, etc.)
                concatenate_messages (bool): Whether to concatenate messages based on length rules.
                max_length (int): Maximum length for concatenated message groups (default: 1000).
                time_window_minutes (int): Time window in minutes to group messages together (default: 30).
                overlap_messages (int): Number of messages to overlap between consecutive groups (default: 2).
        """
        docs: list[Document] = []
        max_count = load_kwargs.get("max_count", 1000)
        wechat_export_dir = load_kwargs.get("wechat_export_dir", None)
        include_non_text = load_kwargs.get("include_non_text", False)
        concatenate_messages = load_kwargs.get("concatenate_messages", False)
        max_length = load_kwargs.get("max_length", 1000)
        time_window_minutes = load_kwargs.get("time_window_minutes", 30)

        # Default WeChat export path
        if wechat_export_dir is None:
            wechat_export_dir = "./wechat_export_test"

        if not os.path.exists(wechat_export_dir):
            print(f"WeChat export directory not found at: {wechat_export_dir}")
            return docs

        try:
            # Find all JSON files in the export directory
            json_files = list(Path(wechat_export_dir).glob("*.json"))
            print(f"Found {len(json_files)} WeChat chat history files")

            count = 0
            for json_file in json_files:
                if count >= max_count and max_count > 0:
                    break

                try:
                    with open(json_file, encoding="utf-8") as f:
                        chat_data = json.load(f)

                    # Extract contact name from filename
                    contact_name = json_file.stem

                    if concatenate_messages:
                        # Filter messages to only include readable text messages
                        readable_messages = []
                        for message in chat_data:
                            try:
                                content = message.get("content", "")
                                if not include_non_text and not self._is_text_message(content):
                                    continue

                                readable_text = self._extract_readable_text(content)
                                if not readable_text and not include_non_text:
                                    continue

                                readable_messages.append(message)
                            except Exception as e:
                                print(f"Error processing message in {json_file}: {e}")
                                continue

                        # Concatenate messages based on rules
                        message_groups = self._concatenate_messages(
                            readable_messages,
                            max_length=max_length,
                            time_window_minutes=time_window_minutes,
                            overlap_messages=0,  # No overlap between groups
                        )

                        # Create documents from concatenated groups
                        for message_group in message_groups:
                            if count >= max_count and max_count > 0:
                                break

                            doc_content, contact_name = self._create_concatenated_content(
                                message_group, contact_name
                            )
                            doc = Document(
                                text=doc_content,
                                metadata={"contact_name": contact_name},
                            )
                            docs.append(doc)
                            count += 1

                        print(
                            f"Created {len(message_groups)} concatenated message groups for {contact_name}"
                        )

                    else:
                        # Original single-message processing
                        for message in chat_data:
                            if count >= max_count and max_count > 0:
                                break

                            # Extract message information
                            message.get("fromUser", "")
                            message.get("toUser", "")
                            content = message.get("content", "")
                            message_text = message.get("message", "")
                            create_time = message.get("createTime", 0)
                            is_sent_from_self = message.get("isSentFromSelf", False)

                            # Handle content that might be dict or string
                            try:
                                # Check if this is a readable text message
                                if not include_non_text and not self._is_text_message(content):
                                    continue

                                # Extract readable text
                                readable_text = self._extract_readable_text(content)
                                if not readable_text and not include_non_text:
                                    continue
                            except Exception as e:
                                # Skip messages that cause processing errors
                                print(f"Error processing message in {json_file}: {e}")
                                continue

                            # Convert timestamp to readable format
                            if create_time:
                                try:
                                    timestamp = datetime.fromtimestamp(create_time)
                                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                                except (ValueError, OSError):
                                    time_str = str(create_time)
                            else:
                                time_str = "Unknown"

                            # Create document content with metadata header and contact info
                            doc_content = f"""
Contact: {contact_name}
Is sent from self: {is_sent_from_self}
Time: {time_str}
Message: {readable_text if readable_text else message_text}
"""

                            # Create document with embedded metadata
                            doc = Document(
                                text=doc_content, metadata={"contact_name": contact_name}
                            )
                            docs.append(doc)
                            count += 1

                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
                    continue

            print(f"Loaded {len(docs)} WeChat chat documents")

        except Exception as e:
            print(f"Error reading WeChat history: {e}")
            return docs

        return docs

    @staticmethod
    def find_wechat_export_dirs() -> list[Path]:
        """
        Find all WeChat export directories.

        Returns:
            List of Path objects pointing to WeChat export directories
        """
        export_dirs = []

        # Look for common export directory names
        possible_dirs = [
            Path("./wechat_export"),
            Path("./wechat_export_direct"),
            Path("./wechat_chat_history"),
            Path("./chat_export"),
        ]

        for export_dir in possible_dirs:
            if export_dir.exists() and export_dir.is_dir():
                json_files = list(export_dir.glob("*.json"))
                if json_files:
                    export_dirs.append(export_dir)
                    print(
                        f"Found WeChat export directory: {export_dir} with {len(json_files)} files"
                    )

        print(f"Found {len(export_dirs)} WeChat export directories")
        return export_dirs

    @staticmethod
    def export_chat_to_file(
        output_file: str = "wechat_chat_export.txt",
        max_count: int = 1000,
        export_dir: str | None = None,
        include_non_text: bool = False,
    ):
        """
        Export WeChat chat history to a text file.

        Args:
            output_file: Path to the output file
            max_count: Maximum number of entries to export
            export_dir: Directory containing WeChat JSON files
            include_non_text: Whether to include non-text messages
        """
        if export_dir is None:
            export_dir = "./wechat_export_test"

        if not os.path.exists(export_dir):
            print(f"WeChat export directory not found at: {export_dir}")
            return

        try:
            json_files = list(Path(export_dir).glob("*.json"))

            with open(output_file, "w", encoding="utf-8") as f:
                count = 0
                for json_file in json_files:
                    if count >= max_count and max_count > 0:
                        break

                    try:
                        with open(json_file, encoding="utf-8") as json_f:
                            chat_data = json.load(json_f)

                        contact_name = json_file.stem
                        f.write(f"\n=== Chat with {contact_name} ===\n")

                        for message in chat_data:
                            if count >= max_count and max_count > 0:
                                break

                            from_user = message.get("fromUser", "")
                            content = message.get("content", "")
                            message_text = message.get("message", "")
                            create_time = message.get("createTime", 0)

                            # Skip non-text messages unless requested
                            if not include_non_text:
                                reader = WeChatHistoryReader()
                                if not reader._is_text_message(content):
                                    continue
                                readable_text = reader._extract_readable_text(content)
                                if not readable_text:
                                    continue
                                message_text = readable_text

                            if create_time:
                                try:
                                    timestamp = datetime.fromtimestamp(create_time)
                                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                                except (ValueError, OSError):
                                    time_str = str(create_time)
                            else:
                                time_str = "Unknown"

                            f.write(f"[{time_str}] {from_user}: {message_text}\n")
                            count += 1

                    except Exception as e:
                        print(f"Error processing {json_file}: {e}")
                        continue

            print(f"Exported {count} chat entries to {output_file}")

        except Exception as e:
            print(f"Error exporting WeChat chat history: {e}")

    def export_wechat_chat_history(self, export_dir: str = "./wechat_export_direct") -> Path | None:
        """
        Export WeChat chat history using wechat-exporter tool.

        Args:
            export_dir: Directory to save exported chat history

        Returns:
            Path to export directory if successful, None otherwise
        """
        try:
            import subprocess
            import sys

            # Create export directory
            export_path = Path(export_dir)
            export_path.mkdir(exist_ok=True)

            print(f"Exporting WeChat chat history to {export_path}...")

            # Check if wechat-exporter directory exists
            if not self.wechat_exporter_dir.exists():
                print(f"wechat-exporter directory not found at: {self.wechat_exporter_dir}")
                return None

            # Install requirements if needed
            requirements_file = self.wechat_exporter_dir / "requirements.txt"
            if requirements_file.exists():
                print("Installing wechat-exporter requirements...")
                subprocess.run(["uv", "pip", "install", "-r", str(requirements_file)], check=True)

            # Run the export command
            print("Running wechat-exporter...")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.wechat_exporter_dir / "main.py"),
                    "export-all",
                    str(export_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            print("Export command output:")
            print(result.stdout)
            if result.stderr:
                print("Export errors:")
                print(result.stderr)

            # Check if export was successful
            if export_path.exists() and any(export_path.glob("*.json")):
                json_files = list(export_path.glob("*.json"))
                print(
                    f"Successfully exported {len(json_files)} chat history files to {export_path}"
                )
                return export_path
            else:
                print("Export completed but no JSON files found")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Export command failed: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Command errors: {e.stderr}")
            return None
        except Exception as e:
            print(f"Export failed: {e}")
            print("Please ensure WeChat is running and WeChatTweak is installed.")
            return None

    def find_or_export_wechat_data(self, export_dir: str = "./wechat_export_direct") -> list[Path]:
        """
        Find existing WeChat exports or create new ones.

        Args:
            export_dir: Directory to save exported chat history if needed

        Returns:
            List of Path objects pointing to WeChat export directories
        """
        export_dirs = []

        # Look for existing exports in common locations
        possible_export_dirs = [
            Path("./wechat_database_export"),
            Path("./wechat_export_test"),
            Path("./wechat_export"),
            Path("./wechat_export_direct"),
            Path("./wechat_chat_history"),
            Path("./chat_export"),
        ]

        for export_dir_path in possible_export_dirs:
            if export_dir_path.exists() and any(export_dir_path.glob("*.json")):
                export_dirs.append(export_dir_path)
                print(f"Found existing export: {export_dir_path}")

        # If no existing exports, try to export automatically
        if not export_dirs:
            print("No existing WeChat exports found. Starting direct export...")

            # Try to export using wechat-exporter
            exported_path = self.export_wechat_chat_history(export_dir)
            if exported_path:
                export_dirs = [exported_path]
            else:
                print(
                    "Failed to export WeChat data. Please ensure WeChat is running and WeChatTweak is installed."
                )

        return export_dirs
