"""
Smoke Test - Verify all key packages can be imported

This test ensures that the basic project structure is correct
and all major modules can be imported without errors.
"""

import pytest


def test_import_mcp_server():
    """Test that mcp_server package can be imported"""
    import src.mcp_server  # noqa: F401
    assert True


def test_import_core():
    """Test that core package can be imported"""
    import src.core  # noqa: F401
    assert True


def test_import_core_settings():
    """Test that core.settings module can be imported"""
    from src.core import settings  # noqa: F401
    assert True


def test_import_ingestion():
    """Test that ingestion package can be imported"""
    import src.ingestion  # noqa: F401
    assert True


def test_import_libs():
    """Test that libs package can be imported"""
    import src.libs  # noqa: F401
    assert True


def test_import_observability():
    """Test that observability package can be imported"""
    import src.observability  # noqa: F401
    assert True


def test_import_observability_logger():
    """Test that observability.logger module can be imported"""
    from src.observability import logger  # noqa: F401
    assert True


def test_smoke_all_core_packages():
    """Comprehensive test: import all critical top-level packages"""
    import src.mcp_server
    import src.core
    import src.ingestion
    import src.libs
    import src.observability

    # Verify key modules exist
    from src.core.settings import Settings, load_settings, validate_settings  # noqa: F401
    from src.observability.logger import get_logger  # noqa: F401

    assert True
