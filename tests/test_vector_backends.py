from __future__ import annotations

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.handlers import search as search_mod
from mmrag.vector_backends import QdrantBackend, SqliteVecBackend


def test_vector_backend_defaults_to_sqlite(isolated_data_dir):
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir))
    assert isinstance(search_mod._vector_backend(), SqliteVecBackend)


def test_vector_backend_can_select_qdrant(isolated_data_dir):
    reset_settings_for_tests(
        Settings(data_dir=isolated_data_dir, vector_backend="qdrant", qdrant_url="http://qdrant")
    )
    backend = search_mod._vector_backend()
    assert isinstance(backend, QdrantBackend)
    assert backend.url == "http://qdrant"
