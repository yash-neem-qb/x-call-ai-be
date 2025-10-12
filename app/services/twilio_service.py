"""
Twilio service for handling phone calls and number validation.
Provides functionality for making outbound calls and validating phone numbers.
"""

import logging
from typing import Optional
from twilio.rest import Client
from twilio.base.exceptions import TwilioException

from app.config.settings import settings

logger = logging.getLogger(__name__)


class TwilioService:
    """Service class for Twilio operations."""
    
    def __init__(self):
        """Initialize Twilio client."""
        self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    
    async def check_number_allowed(self, phone_number: str) -> bool:
        """
        Check if a phone number is allowed to be called.
        
        Args:
            phone_number: The phone number to check
            
        Returns:
            bool: True if the number is allowed, False otherwise
        """
        try:
            # Check if it's an incoming phone number
            incoming_numbers = self.client.incoming_phone_numbers.list(phone_number=phone_number)
            if incoming_numbers:
                logger.info(f"Number {phone_number} is a valid incoming phone number")
                return True
            
            # Check if it's a verified outgoing caller ID
            outgoing_caller_ids = self.client.outgoing_caller_ids.list(phone_number=phone_number)
            if outgoing_caller_ids:
                logger.info(f"Number {phone_number} is a verified outgoing caller ID")
                return True
            
            logger.warning(f"Number {phone_number} is not recognized as a valid number")
            return False
            
        except TwilioException as e:
            logger.error(f"Twilio error checking phone number {phone_number}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking phone number {phone_number}: {e}")
            return False
    
    async def make_call(self, phone_number: str, status_callback_url: str = None) -> str:
        """
        Make an outbound call to the specified phone number.
        
        Args:
            phone_number: The phone number to call (must be in E.164 format)
            status_callback_url: Optional URL for status callbacks
            
        Returns:
            str: The call SID
            
        Raises:
            ValueError: If the phone number is invalid or not allowed
            TwilioException: If there's an error with the Twilio API
        """
        if not phone_number:
            raise ValueError("Phone number is required")
        
        # Validate the phone number is allowed
        is_allowed = await self.check_number_allowed(phone_number)
        if not is_allowed:
            raise ValueError(
                f"The number {phone_number} is not recognized as a valid outgoing number "
                "or caller ID. Please ensure you have permission to call this number."
            )
        
        # Create TwiML for the call
        twiml = self._create_call_twiml()
        
        # Set up status callback URL
        callback_url = f"https://{settings.domain}{status_callback_url}" if status_callback_url else None
        
        try:
            # Prepare call parameters
            call_params = {
                'from_': settings.phone_number_from,
                'to': phone_number,
                'twiml': twiml
            }
            
            # Add status callbacks if URL provided
            if callback_url:
                call_params.update({
                    'status_callback': callback_url,
                    'status_callback_event': ['initiated', 'ringing', 'answered', 'completed'],
                    'status_callback_method': 'POST'
                })
            
            # Make the call
            call = self.client.calls.create(**call_params)
            
            logger.info(f"Call initiated successfully. Call SID: {call.sid}")
            return call.sid
            
        except TwilioException as e:
            logger.error(f"Twilio error making call to {phone_number}: {e}")
            raise ValueError(f"Failed to make call: {e}")
        except Exception as e:
            logger.error(f"Unexpected error making call to {phone_number}: {e}")
            raise ValueError(f"Failed to make call: {e}")
    
    def _create_call_twiml(self) -> str:
        """
        Create TwiML for the outbound call.
        
        Returns:
            str: The TwiML XML string
        """
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response>'
            f'<Connect>'
            f'<Stream url="wss://{settings.domain}/api/v1/websocket/media-stream" />'
            f'</Connect>'
            f'<Say>Thank you for calling. Goodbye.</Say>'
            f'</Response>'
        )
    
    async def log_call_sid(self, call_sid: str) -> None:
        """
        Log the call SID for tracking purposes.
        
        Args:
            call_sid: The call SID to log
        """
        logger.info(f"Call started with SID: {call_sid}")
    
    async def configure_inbound_webhooks(self, 
                                       account_sid: str, 
                                       auth_token: str, 
                                       phone_number: str,
                                       webhook_base_url: str) -> dict:
        """
        Configure Twilio webhooks for inbound calls.
        
        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            phone_number: The phone number to configure (in E.164 format)
            webhook_base_url: Base URL for webhooks (e.g., https://your-domain.com)
            
        Returns:
            dict: Configuration result with success status and details
            
        Raises:
            ValueError: If configuration fails
            TwilioException: If there's an error with the Twilio API
        """
        try:
            # Create a temporary client with provided credentials
            temp_client = Client(account_sid, auth_token)
            
            # Find the incoming phone number
            incoming_numbers = temp_client.incoming_phone_numbers.list(phone_number=phone_number)
            
            if not incoming_numbers:
                raise ValueError(f"Phone number {phone_number} not found in your Twilio account")
            
            incoming_number = incoming_numbers[0]
            
            # Configure webhook URLs
            webhook_url = f"{webhook_base_url}/api/v1/inbound/webhook"
            status_callback_url = f"{webhook_base_url}/api/v1/inbound/status"
            
            # Update the incoming phone number with webhook configuration
            updated_number = temp_client.incoming_phone_numbers(incoming_number.sid).update(
                voice_url=webhook_url,
                voice_method='POST',
                status_callback=status_callback_url,
                status_callback_method='POST'
            )
            
            logger.info(f"Successfully configured webhooks for {phone_number}")
            logger.info(f"Voice URL: {webhook_url}")
            logger.info(f"Status Callback URL: {status_callback_url}")
            
            return {
                "success": True,
                "phone_number": phone_number,
                "voice_url": webhook_url,
                "status_callback_url": status_callback_url,
                "friendly_name": updated_number.friendly_name,
                "sid": updated_number.sid
            }
            
        except TwilioException as e:
            logger.error(f"Twilio error configuring webhooks for {phone_number}: {e}")
            raise ValueError(f"Failed to configure webhooks: {e}")
        except Exception as e:
            logger.error(f"Unexpected error configuring webhooks for {phone_number}: {e}")
            raise ValueError(f"Failed to configure webhooks: {e}")


# Global Twilio service instance
twilio_service = TwilioService()
