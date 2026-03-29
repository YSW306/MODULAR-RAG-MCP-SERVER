#!/usr/bin/env python3
"""
Modular RAG MCP Server - Main Entry Point

This is the main entry point for the MCP (Model Context Protocol) server.
It handles initialization, configuration loading, and server startup.
"""

import sys
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.settings import load_settings, validate_settings
from observability.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    Main entry point for the MCP server.

    Steps:
    1. Load configuration from config/settings.yaml
    2. Validate configuration
    3. Initialize MCP server (to be implemented in Phase E)
    """
    try:
        # Load settings
        logger.info("Loading configuration from config/settings.yaml")
        settings = load_settings()

        # Validate settings
        logger.info("Validating configuration")
        validate_settings(settings)

        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  LLM Provider: {settings.llm.provider}/{settings.llm.model}")
        logger.info(f"  Embedding Provider: {settings.embedding.provider}")

        # TODO: Initialize and start MCP server (Phase E)
        logger.info("✓ MCP Server initialized (Phase E - to be implemented)")

        return 0

    except Exception as e:
        logger.error(f"✗ Failed to initialize server: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
