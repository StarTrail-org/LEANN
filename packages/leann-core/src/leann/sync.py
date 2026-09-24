import logging
import os
import pickle
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Keep in sync with leann build's default extension allowlist (load_documents).
DEFAULT_INDEX_EXTENSIONS: list[str] = [
    ".txt",
    ".md",
    ".docx",
    ".pptx",
    ".pdf",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".vue",
    ".svelte",
    ".ipynb",
    ".R",
    ".jl",
]


def hash_data(data: str | bytes):
    if isinstance(data, str):
        data = data.encode()
    return sha256(data).hexdigest()


def parse_include_extensions(custom_file_types: Optional[str]) -> list[str]:
    """Return the extension allowlist used for sync/watch (matches build defaults)."""
    if not custom_file_types:
        return list(DEFAULT_INDEX_EXTENSIONS)
    extensions = [ext.strip() for ext in custom_file_types.split(",") if ext.strip()]
    return [ext if ext.startswith(".") else f".{ext}" for ext in extensions]


def _extension_allowed(path: Path, include_extensions: list[str]) -> bool:
    allowed = {ext.lower() for ext in include_extensions}
    return path.suffix.lower() in allowed


def _path_has_hidden_segment(path: Path) -> bool:
    return any(part.startswith(".") and part not in (".", "..") for part in path.parts)


def _hash_file_bytes(path: Path) -> str:
    with open(path, "rb") as f:
        return hash_data(f.read())


def _iter_directory_files(
    root_dir: str,
    include_extensions: list[str],
    include_hidden: bool,
) -> list[str]:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        return []

    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        current = Path(dirpath)
        for name in filenames:
            if not include_hidden and name.startswith("."):
                continue
            file_path = (current / name).resolve()
            try:
                rel = file_path.relative_to(root)
            except ValueError:
                continue
            if not include_hidden and _path_has_hidden_segment(rel):
                continue
            if not _extension_allowed(file_path, include_extensions):
                continue
            if file_path.is_file():
                paths.append(str(file_path))
    return paths


@dataclass
class MerkleTreeNode:
    ## TODO: this merkle tree only has two layer, need to improve if we want to scale to large codebase
    hash: str
    data: str
    children: dict[str, "MerkleTreeNode"] = field(default_factory=dict)
    parent_id: str | None = None


class MerkleTree:
    def __init__(self):
        self.nodes: dict[str, MerkleTreeNode] = {}
        self.root: MerkleTreeNode | None = None

    def add_node(self, data: str, parent_id=None, hash: Optional[str] = None):
        hash = hash_data(data) if hash is None else hash

        node = MerkleTreeNode(hash=hash, data=data, parent_id=parent_id)
        self.nodes[hash] = node

        if parent_id is None:
            self.root = node
        else:
            self.nodes[parent_id].children[hash] = node

        return hash

    def compare_with(self, other: "MerkleTree"):
        """
        Simple comparison of two flat trees. Check the individual file hashes
        only if the root has changed, otherwise return no changes.
        """
        assert self.root is not None and other.root is not None

        if self.root.hash == other.root.hash:
            return [], [], []

        old_files = self.root.children
        new_files = other.root.children

        all_nodes = new_files.keys() | old_files.keys()

        added, removed, modified = [], [], []
        for path in all_nodes:
            if path in new_files and path in old_files:
                if new_files[path].data != old_files[path].data:
                    modified.append(path)
            elif path in new_files and path not in old_files:
                added.append(path)
            else:
                removed.append(path)

        return added, removed, modified


class FileSynchronizer:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        explicit_files: Optional[list[str]] = None,
        ignore_patterns: Optional[list] = None,
        include_extensions: Optional[list[str]] = None,
        include_hidden: bool = False,
        auto_load=True,
        snapshot_path: Optional[str] = None,
    ):
        self.root_dir = str(Path(root_dir).resolve()) if root_dir else None
        self.explicit_files = (
            [str(Path(path).resolve()) for path in explicit_files] if explicit_files else []
        )
        if self.root_dir is None and not self.explicit_files:
            raise ValueError("FileSynchronizer requires root_dir and/or explicit_files")
        if self.root_dir is not None and not os.path.isdir(self.root_dir):
            raise ValueError("This is not a valid directory")

        self.ignore_patterns = ignore_patterns
        self.include_extensions = include_extensions or list(DEFAULT_INDEX_EXTENSIONS)
        self.include_hidden = include_hidden
        self._custom_snapshot_path = snapshot_path
        self._pending_tree: Optional[MerkleTree] = None
        self.tree: Optional[MerkleTree] = None
        if auto_load:
            self.load_snapshot()

    def _collect_paths(self) -> list[str]:
        paths: list[str] = []
        if self.root_dir:
            paths.extend(
                _iter_directory_files(
                    self.root_dir,
                    self.include_extensions,
                    self.include_hidden,
                )
            )
        for file_path in self.explicit_files:
            path = Path(file_path).resolve()
            if not path.is_file():
                continue
            # A file named on its own is judged by its own name, not by its
            # ancestors: pointing at a directory already overrides their hidden-ness.
            if not self.include_hidden and path.name.startswith("."):
                continue
            if not _extension_allowed(path, self.include_extensions):
                continue
            paths.append(str(path))
        return sorted(set(paths))

    def generate_file_hashes(self):
        file_hashes: dict[str, str] = {}
        for file_path in self._collect_paths():
            try:
                file_hashes[file_path] = _hash_file_bytes(Path(file_path))
            except OSError:
                logger.warning("Cannot hash file %s", file_path)
        return file_hashes

    def build_merkle_tree(self, file_hashes):
        """
        Build a flat merkle tree suitable for quick checking of file changes.
        """
        tree = MerkleTree()

        sorted_paths = sorted(file_hashes)
        root_data = "".join(path + file_hashes[path] for path in sorted_paths)

        root_id = tree.add_node(root_data)

        for path in sorted_paths:
            tree.add_node(file_hashes[path], parent_id=root_id, hash=path)

        return tree

    def detect_changes(self) -> tuple[list[str], list[str], list[str]]:
        """Detect changes without persisting. Call commit() after successful processing."""
        file_hashes = self.generate_file_hashes()
        new_tree = self.build_merkle_tree(file_hashes)
        self._pending_tree = new_tree

        if self.tree is None:
            return list(file_hashes.keys()), [], []

        return self.tree.compare_with(new_tree)

    def commit(self):
        """Persist the pending snapshot after successful processing."""
        if self._pending_tree is not None:
            self.tree = self._pending_tree
            self._pending_tree = None
            self.save_snapshot()

    def create_snapshot(self):
        """Build and persist a snapshot from the current file state (for initial / forced builds)."""
        file_hashes = self.generate_file_hashes()
        self.tree = self.build_merkle_tree(file_hashes)
        self.save_snapshot()

    def check_for_changes(self) -> tuple[list[str], list[str], list[str]]:
        """Detect and auto-commit changes (convenience wrapper)."""
        changes = self.detect_changes()
        self.commit()
        return changes

    @property
    def snapshot_path(self):
        if self._custom_snapshot_path:
            return self._custom_snapshot_path
        base = self.root_dir or "files"
        return f"{base}.sync_context.pickle"

    def save_snapshot(self):
        assert self.tree is not None

        with open(self.snapshot_path, "wb") as f:
            pickle.dump(self.tree, f)

    def load_snapshot(self):
        try:
            with open(self.snapshot_path, "rb") as f:
                self.tree = pickle.load(f)
        except FileNotFoundError:
            self.tree = None
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            ImportError,
            IndexError,
        ) as exc:
            logger.warning("Cannot load sync snapshot %s: %s", self.snapshot_path, exc)
            self.tree = None
