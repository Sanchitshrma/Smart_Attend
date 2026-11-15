import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.fake_supabase import FakeSupabase  # noqa: E402


@pytest.fixture()
def fake_db():
    from src.database import config

    from src.database import db
    fake = FakeSupabase()
    config.set_client_for_testing(fake)
    db._TEMPLATES_AVAILABLE = None  # re-detect migration 010 for every test
    from src.pipelines.face_pipeline import clear_gallery_cache
    clear_gallery_cache()
    yield fake
    clear_gallery_cache()
    db._TEMPLATES_AVAILABLE = None
    config.set_client_for_testing(None)
