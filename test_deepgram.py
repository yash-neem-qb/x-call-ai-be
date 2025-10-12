"""
Test script for Deepgram API connectivity
"""

import os
import asyncio
import websockets
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API key from .env file
from dotenv import load_dotenv
load_dotenv()

# Get Deepgram API key
api_key = os.environ.get('DEEPGRAM_API_KEY')
if not api_key:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

logger.info(f"Using API key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")

async def test_deepgram_connection():
    """Test connection to Deepgram API"""
    
    # Deepgram WebSocket URL
    base_url = "wss://api.deepgram.com/v1/listen"
    
    # Basic params for testing
    params = {
        "model": "nova-2",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "language": "en"
    }
    
    # Build query string
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    deepgram_url = f"{base_url}?{query_string}"
    
    logger.info(f"Connecting to Deepgram: {deepgram_url}")
    
    # Try with lowercase 'token'
    try:
        logger.info("Trying with lowercase 'token'...")
        deepgram_ws = await asyncio.wait_for(
            websockets.connect(
                deepgram_url,
                extra_headers={"Authorization": f"token {api_key}"}
            ),
            timeout=30.0
        )
        logger.info("✅ Successfully connected with lowercase 'token'")
        await deepgram_ws.close()
        return True
    except Exception as e:
        logger.error(f"❌ Failed with lowercase 'token': {e}")
    
    # Try with uppercase 'Token'
    try:
        logger.info("Trying with uppercase 'Token'...")
        deepgram_ws = await asyncio.wait_for(
            websockets.connect(
                deepgram_url,
                extra_headers={"Authorization": f"Token {api_key}"}
            ),
            timeout=30.0
        )
        logger.info("✅ Successfully connected with uppercase 'Token'")
        await deepgram_ws.close()
        return True
    except Exception as e:
        logger.error(f"❌ Failed with uppercase 'Token': {e}")
    
    # Try with Bearer
    try:
        logger.info("Trying with Bearer...")
        deepgram_ws = await asyncio.wait_for(
            websockets.connect(
                deepgram_url,
                extra_headers={"Authorization": f"Bearer {api_key}"}
            ),
            timeout=30.0
        )
        logger.info("✅ Successfully connected with Bearer")
        await deepgram_ws.close()
        return True
    except Exception as e:
        logger.error(f"❌ Failed with Bearer: {e}")
    
    logger.error("All connection attempts failed")
    return False

if __name__ == "__main__":
    asyncio.run(test_deepgram_connection())
