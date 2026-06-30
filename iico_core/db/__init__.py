"""
iico_core/db/__init__.py
=========================
Base de datos ultra-ligera para notas, chunks y embeddings.
"""

from .note_db import NoteDB
from .watcher import NoteWatcher, NoteParser, SyncReport

__all__ = ["NoteDB", "NoteWatcher", "NoteParser", "SyncReport"]
