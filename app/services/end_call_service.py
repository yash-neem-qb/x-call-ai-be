"""
Built-in End Call Service.
Provides functionality to detect when users want to end the call and handle graceful termination.
"""

import logging
import re
from typing import Optional, Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)


class EndCallService:
    """
    Service for detecting and handling end call requests.
    
    This service provides built-in functionality to detect when users want to end
    the call through natural language and handle graceful call termination.
    """
    
    def __init__(self):
        """Initialize the end call service."""
        self.enabled = settings.end_call_enabled
        self.end_phrases = [phrase.lower() for phrase in settings.end_call_phrases]
        self.end_reason = settings.end_call_end_reason
        
        logger.info(f"End call service initialized - Enabled: {self.enabled}, Phrases: {len(self.end_phrases)}")
    
    def should_end_call(self, user_input: str) -> bool:
        """
        Check if the user input indicates they want to end the call.
        
        Args:
            user_input: The user's input text
            
        Returns:
            bool: True if the user wants to end the call
        """
        if not self.enabled:
            return False
        
        try:
            # Normalize input for comparison
            normalized_input = user_input.lower().strip()
            
            # Remove punctuation for better matching
            clean_input = re.sub(r'[^\w\s]', ' ', normalized_input)
            clean_input = ' '.join(clean_input.split())  # Remove extra spaces
            
            # Check for exact phrase matches
            for phrase in self.end_phrases:
                if phrase in clean_input:
                    logger.info(f"End call detected - Phrase: '{phrase}' in input: '{user_input[:50]}...'")
                    return True
            
            # Check for standalone end call words
            end_words = ["bye", "goodbye", "farewell", "disconnect", "hangup", "hang up"]
            words = clean_input.split()
            
            # If input is very short and contains end words, likely an end request
            if len(words) <= 3:
                for word in words:
                    if word in end_words:
                        logger.info(f"End call detected - End word: '{word}' in short input: '{user_input}'")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking end call request: {e}")
            return False
    
    def get_end_call_response(self, user_input: str) -> str:
        """
        Get an appropriate response when ending the call.
        
        Args:
            user_input: The user's input that triggered the end call
            
        Returns:
            str: Appropriate goodbye response
        """
        try:
            # Determine response based on user's input
            normalized_input = user_input.lower()
            
            if any(word in normalized_input for word in ["thank", "thanks"]):
                return "You're very welcome! Have a great day and feel free to call again anytime. Goodbye!"
            elif any(word in normalized_input for word in ["goodbye", "bye"]):
                return "Goodbye! It was great talking with you. Have a wonderful day!"
            elif any(word in normalized_input for word in ["good day", "have a good"]):
                return "Thank you! You have a wonderful day too. Goodbye!"
            elif any(word in normalized_input for word in ["later", "see you"]):
                return "Sounds good! Talk to you later. Take care!"
            else:
                return "Thank you for calling! Have a great day and goodbye!"
                
        except Exception as e:
            logger.error(f"Error generating end call response: {e}")
            return "Thank you for calling! Goodbye!"
    
    def get_end_call_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the end call event.
        
        Returns:
            Dict containing end call metadata
        """
        return {
            "end_reason": self.end_reason,
            "end_type": "user_requested",
            "service": "built_in_end_call",
            "enabled": self.enabled,
            "phrases_count": len(self.end_phrases)
        }


# Global instance
end_call_service = EndCallService()
