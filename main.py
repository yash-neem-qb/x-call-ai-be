#!/usr/bin/env python3
"""
Main entry point for the AI Voice Assistant application.
Handles command line arguments and starts the FastAPI server.
"""

import argparse
import logging
import uvicorn
from dotenv import load_dotenv

from app.core.app import app
from app.config.settings import settings
from app.core.logging import setup_logging
# Removed old call_handler import - using unified_pipeline_manager now

# Load environment variables
load_dotenv()

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI Voice Assistant"
    )
    parser.add_argument(
        "--call",
        type=str,
        help="Phone number to call (must be in E.164 format, e.g., +1234567890)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind the server to (default: {settings.port})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Display legal compliance notice
    print(
        'Our recommendation is to always disclose the use of AI for outbound or inbound calls.\n'
        'Reminder: All of the rules of TCPA apply even if a call is made by AI.\n'
        'Check with your counsel for legal and compliance advice.\n'
    )
    
    # Start the server
    logger.info(f"Starting AI Voice Assistant server on {args.host}:{args.port}")
    uvicorn.run(
        "app.core.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()