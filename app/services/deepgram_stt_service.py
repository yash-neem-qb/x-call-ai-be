"""
Speech-to-Text service using Deepgram.
Provides a clean interface for real-time transcription of audio to text.
"""

import json
import logging
import asyncio
import websockets
import base64
import uuid
from typing import Optional, Dict, Any, Callable, Coroutine, List, Union

from app.config.settings import settings

logger = logging.getLogger(__name__)


class DeepgramSTTService:
    """Service for Speech-to-Text operations using Deepgram API."""
    
    def __init__(self):
        """Initialize the Deepgram STT service."""
        self.api_key = settings.deepgram_api_key
        # No hardcoded defaults - everything comes from database
        
        # Active connections store
        self.active_connections = {}
        
        # Deepgram WebSocket URL
        self.base_url = "wss://api.deepgram.com/v1/listen"
    
    async def initialize(self):
        """
        Initialize the STT service.
        This is a no-op for Deepgram as connections are per-session.
        """
        logger.info("Deepgram STT service initialized")
        return True
    
    async def create_transcription_session(self, session_id: str, 
                                          on_transcription_delta: Callable[[str, str, float], Coroutine],
                                          on_transcription_complete: Callable[[str, str], Coroutine],
                                          model: str,
                                          language: str,
                                          on_speech_started: Optional[Callable[[str], Coroutine]] = None,
                                          is_web_stream: bool = False) -> bool:
        """
        Create a new transcription session.
        
        Args:
            session_id: Unique identifier for this session
            on_transcription_delta: Callback for real-time transcription updates
            on_transcription_complete: Callback for final transcription
            model: Model name to use (required)
            language: Language code (required)
            on_speech_started: Optional callback when speech is detected
            is_web_stream: Whether this is a web-based stream (16kHz PCM) vs Twilio (8kHz mulaw)
            
        Returns:
            bool: True if session was created successfully
        """
        try:
            # Validate required parameters
            if not model:
                raise ValueError("Model is required")
            if not language:
                raise ValueError("Language is required")

            # Deepgram expects `en` for many nova-3 variants over WS; normalize if needed
            transcription_language = language
            if model.startswith("nova-3") and language.lower().startswith("en-us"):
                logger.info("Normalizing Deepgram language from en-US to en for nova-3")
                transcription_language = "en"
            
            logger.info(f"Creating Deepgram transcription session with model: {model}, language: {transcription_language}")
            
            # Build query parameters according to Deepgram API documentation
            # Use best options for each model type with noise suppression
            # Configure audio format based on stream type
            if is_web_stream:
                # Web-based streams use 16kHz PCM
                encoding = "linear16"
                sample_rate = "16000"
            else:
                # Twilio streams use 8kHz mulaw
                encoding = "mulaw"
                sample_rate = "8000"
            
            # Validate model is supported
            supported_models = [
                "nova-2", "nova-3", 
                "nova-3-general", "nova-3-medical",
                "nova-2-general", "nova-2-meeting", "nova-2-phonecall", 
                "nova-2-finance", "nova-2-conversational-ai", "nova-2-voicemail",
                "nova-2-video", "nova-2-medical", "nova-2-drive-thru", "nova-2-automotive"
            ]
            if model not in supported_models:
                raise ValueError(f"Model {model} is not supported. Supported models: {supported_models}")
            
            # Use a minimal set of parameters that we know work from our test_deepgram.py
            # This ensures basic connectivity, we can add more parameters once basic connection works
            params = {
                "model": model,
                "encoding": encoding,
                "sample_rate": sample_rate,
                "channels": "1",
                "language": transcription_language
            }
            
            # For debugging only - add these once connection is stable
            if False:  # Disable all extra parameters until we have basic connectivity
                params.update({
                    "punctuate": "true",
                    "smart_format": "true",
                    "interim_results": "true",
                    "noise_reduction": "true",
                    "profanity_filter": "false",
                    "inactivity_timeout": "30"
                })
                
                # Add call-specific extras with optimized VAD for reduced background noise
                if model.startswith("nova-2"):
                    # Low-latency call tuned with minimal VAD padding
                    params.update({
                        "filler_words": "true",
                        "diarize": "true",
                        "endpointing": "250",
                        "utterance_end_ms": "600",
                        "vad_events": "true",
                        "vad_turn_padding": "100"
                    })
                elif model.startswith("nova-3"):
                    # Enable best features for nova-3 models with minimal VAD padding
                    params.update({
                        "diarize": "true",
                        "endpointing": "300",
                        "vad_events": "true",
                        "vad_turn_padding": "150"
                    })

            # Log a concise view of what we are actually sending to Deepgram
            features_snapshot = {
                "model": params.get("model"),
                "language": params.get("language"),
                "encoding": params.get("encoding"),
                "sample_rate": params.get("sample_rate"),
                "channels": params.get("channels"),
                # Common
                "punctuate": params.get("punctuate"),
                "smart_format": params.get("smart_format"),
                "interim_results": params.get("interim_results"),
                # Noise suppression
                "noise_reduction": params.get("noise_reduction"),
                # Call extras (only present for nova-2)
                "filler_words": params.get("filler_words"),
                "diarize": params.get("diarize"),
                "endpointing": params.get("endpointing"),
                "utterance_end_ms": params.get("utterance_end_ms"),
                "vad_events": params.get("vad_events"),
                "vad_turn_padding": params.get("vad_turn_padding"),
            }
            logger.debug(f"Deepgram connection parameters: {features_snapshot}")
            
            # Build the URL with query parameters
            query_string = "&".join([f"{key}={value}" for key, value in params.items()])
            deepgram_url = f"{self.base_url}?{query_string}"
            
            logger.info(f"Connecting to Deepgram with model: {model}")
            logger.debug(f"Deepgram URL: {deepgram_url}")
            
            # Connect to Deepgram WebSocket with timeout
            try:
                deepgram_ws = await asyncio.wait_for(
                    websockets.connect(
                        deepgram_url,
                        extra_headers={
                            # Use lowercase 'token' scheme as per our successful test
                            "Authorization": f"token {self.api_key}"
                        }
                    ),
                    timeout=60.0
                )
                logger.info("Successfully connected to Deepgram WebSocket")
                if model.startswith("nova-3"):
                    logger.debug("Deepgram connected with nova-3 features enabled (diarize/endpointing/vad_events)")
                else:
                    logger.debug("Deepgram connected with nova-2 call features enabled (diarize, filler_words, endpointing, utterance_end_ms, vad_events)")
            except Exception as e:
                logger.error(f"Failed to connect to Deepgram (params={params}): {e}")
                # If we're on a nova-3 family, retry once with only the absolute minimum params
                if model.startswith("nova-3"):
                    try:
                        minimal_params = {
                            "model": model,
                            "encoding": encoding,
                            "sample_rate": sample_rate,
                            "channels": "1",
                            "language": transcription_language,
                            "inactivity_timeout": "30",
                        }
                        minimal_qs = "&".join([f"{k}={v}" for k, v in minimal_params.items()])
                        minimal_url = f"{self.base_url}?{minimal_qs}"
                        logger.info("Retrying Deepgram connection with minimal params for nova-3")
                        deepgram_ws = await asyncio.wait_for(
                            websockets.connect(
                                minimal_url,
                                extra_headers={"Authorization": f"token {self.api_key}"}
                            ),
                            timeout=60.0
                        )
                        logger.info("Successfully connected to Deepgram (minimal nova-3 params)")
                        # Overwrite deepgram_url for debugging context
                        deepgram_url = minimal_url
                    except Exception as e2:
                        logger.error(f"Deepgram minimal retry failed (params={minimal_params}): {e2}")
                        return False
                else:
                    return False
            
            # Store connection and callbacks
            self.active_connections[session_id] = {
                "websocket": deepgram_ws,
                "on_transcription_delta": on_transcription_delta,
                "on_transcription_complete": on_transcription_complete,
                "on_speech_started": on_speech_started,
                "task": None,
                "model": model,
                "language": transcription_language,
                "last_activity": asyncio.get_event_loop().time(),
                "metadata": {},
                "is_web_stream": is_web_stream,
                # Store exactly what we used to connect for debugging
                "connect_params": features_snapshot
            }
            
            # Start listening for transcription events
            self.active_connections[session_id]["task"] = asyncio.create_task(
                self._handle_transcription_events(session_id)
            )
            
            logger.info(f"Deepgram transcription session created: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Deepgram transcription session: {e}")
            return False
    
    async def _handle_transcription_events(self, session_id: str):
        """
        Handle transcription events from Deepgram.
        
        Args:
            session_id: The session identifier
        """
        if session_id not in self.active_connections:
            logger.error(f"No active connection for session: {session_id}")
            return
        
        connection = self.active_connections[session_id]
        deepgram_ws = connection["websocket"]
        on_delta = connection["on_transcription_delta"]
        on_complete = connection["on_transcription_complete"]
        
        try:
            async for message in deepgram_ws:
                # Update last activity time
                connection["last_activity"] = asyncio.get_event_loop().time()
                
                # Parse response
                response = json.loads(message)
                response_type = response.get("type")
                
                # Process different response types according to Deepgram documentation
                if response_type == "Results":
                    # Standard transcription results
                    await self._process_results(session_id, response, on_delta, on_complete)
                    
                elif response_type == "Metadata":
                    # Metadata about the audio stream
                    await self._process_metadata(session_id, response)
                    
                elif response_type == "SpeechStarted":
                    # Speech started event
                    logger.debug(f"Speech started detected for session {session_id}")
                    cb = connection.get("on_speech_started")
                    if cb:
                        try:
                            await cb(session_id)
                        except Exception as cb_err:
                            logger.warning(f"on_speech_started callback error: {cb_err}")
                    
                elif response_type == "SpeechFinished" or response_type == "UtteranceEnd":
                    # Speech ended or utterance end event
                    logger.debug(f"Speech ended detected for session {session_id}")
                    
                elif response_type == "Finalize":
                    # Finalize response
                    logger.debug(f"Finalize response received for session {session_id}")
                    
                elif response_type == "CloseStream":
                    # Stream closed by server
                    logger.info(f"Deepgram closed stream for session {session_id}")
                    
                else:
                    # Unknown response type
                    logger.warning(f"Unknown response type from Deepgram: {response_type}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Deepgram WebSocket connection closed for session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Error in Deepgram transcription handler: {e}")
    
    async def _process_results(self, session_id: str, response: Dict[str, Any], 
                              on_delta: Callable[[str, str], Coroutine],
                              on_complete: Callable[[str, str], Coroutine]):
        """
        Process transcription results from Deepgram.
        
        Args:
            session_id: The session identifier
            response: The response from Deepgram
            on_delta: Callback for transcription deltas
            on_complete: Callback for final transcriptions
        """
        try:
            # Extract channel data
            if "channel" in response and "alternatives" in response["channel"]:
                alternatives = response["channel"]["alternatives"]
                if alternatives and "transcript" in alternatives[0]:
                    # Extract transcript and metadata
                    transcript = alternatives[0]["transcript"]
                    confidence = alternatives[0].get("confidence", 0.0)
                    is_final = response.get("is_final", False)
                    speech_final = response.get("speech_final", False)
                    from_finalize = response.get("from_finalize", False)
                    
                    # Process words if available
                    words = []
                    if "words" in alternatives[0]:
                        words = alternatives[0]["words"]
                        
                    # Only process non-empty transcripts
                    if transcript.strip():
                        # Log the transcript
                        status = "FINAL" if is_final else "interim"
                        logger.info(f"Deepgram transcript [{status}]: {transcript[:50]}... (confidence: {confidence:.2f})")
                        
                        # Store metadata in connection
                        if "metadata" in response:
                            self.active_connections[session_id]["metadata"] = response["metadata"]
                        
                        # Send the delta for all transcripts with confidence
                        await on_delta(session_id, transcript, confidence)
                        
                        # Send complete callback for final transcripts
                        if is_final or speech_final or from_finalize:
                            await on_complete(session_id, transcript)
        except Exception as e:
            logger.error(f"Error processing Deepgram results: {e}")
    
    async def _process_metadata(self, session_id: str, response: Dict[str, Any]):
        """
        Process metadata from Deepgram.
        
        Args:
            session_id: The session identifier
            response: The metadata response
        """
        try:
            # Store metadata in connection
            self.active_connections[session_id]["metadata"] = {
                "request_id": response.get("request_id"),
                "created": response.get("created"),
                "duration": response.get("duration"),
                "channels": response.get("channels")
            }
            
            logger.debug(f"Received metadata for session {session_id}: {self.active_connections[session_id]['metadata']}")
        except Exception as e:
            logger.error(f"Error processing Deepgram metadata: {e}")
    
    @staticmethod
    def _ulaw_to_linear16(ulaw_bytes: bytes) -> bytes:
        """Convert G.711 mu-law bytes to 16-bit signed PCM (little-endian)."""
        result = bytearray(len(ulaw_bytes) * 2)
        idx = 0
        for b in ulaw_bytes:
            u = b ^ 0xFF
            sign = u & 0x80
            exponent = (u >> 4) & 0x07
            mantissa = u & 0x0F
            sample = ((mantissa << 4) + 0x08) << exponent
            sample -= 0x84
            if sign:
                sample = -sample
            if sample > 32767:
                sample = 32767
            if sample < -32768:
                sample = -32768
            result[idx] = sample & 0xFF
            result[idx + 1] = (sample >> 8) & 0xFF
            idx += 2
        return bytes(result)
    
    @staticmethod
    def _upsample_linear_interp(pcm_le_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
        """Naive linear interpolation upsample for 16-bit little-endian PCM mono."""
        if out_rate == in_rate:
            return pcm_le_bytes
        import array
        ratio = out_rate / in_rate
        arr = array.array('h')
        arr.frombytes(pcm_le_bytes)
        if ratio.is_integer():
            factor = int(ratio)
            out = array.array('h')
            for s in arr:
                out.extend([s] * factor)
            return out.tobytes()
        out_len = int(len(arr) * ratio)
        out = array.array('h', [0] * out_len)
        for i in range(out_len):
            src_pos = i / ratio
            j = int(src_pos)
            t = src_pos - j
            s0 = arr[j]
            s1 = arr[j + 1] if j + 1 < len(arr) else arr[j]
            out[i] = int((1 - t) * s0 + t * s1)
        return out.tobytes()
    
    async def process_audio(self, session_id: str, audio_data: str) -> bool:
        """
        Process audio data for transcription.
        
        Args:
            session_id: The session identifier
            audio_data: Base64-encoded audio data (mulaw format)
            
        Returns:
            bool: True if audio was processed successfully
        """
        if session_id not in self.active_connections:
            logger.error(f"No active Deepgram connection for session: {session_id}")
            return False
        
        try:
            connection = self.active_connections[session_id]
            deepgram_ws = connection["websocket"]
            
            # Check if connection is open
            if not deepgram_ws.open:
                logger.warning(f"WebSocket connection closed for session: {session_id}")
                return False
            
            # Update last activity time
            connection["last_activity"] = asyncio.get_event_loop().time()
            
            # Decode base64 audio data
            input_bytes = base64.b64decode(audio_data)

            # Handle different audio formats based on stream type
            if connection.get("is_web_stream"):
                # Web streams send 16kHz PCM data directly
                await deepgram_ws.send(input_bytes)
            elif connection.get("transcode"):
                # If nova-3 full transcoding is enabled, convert 8k ulaw -> 16k PCM
                try:
                    pcm8 = self._ulaw_to_linear16(input_bytes)
                    pcm16 = self._upsample_linear_interp(pcm8, 8000, 16000)
                    await deepgram_ws.send(pcm16)
                except Exception as transcode_err:
                    logger.error(f"Deepgram transcoding error: {transcode_err}")
                    return False
            else:
                # Send raw audio data to Deepgram as-is (Twilio mulaw)
                await deepgram_ws.send(input_bytes)
            
            # Send keep-alive message periodically
            if "last_keepalive" not in connection or \
               (asyncio.get_event_loop().time() - connection.get("last_keepalive", 0)) > 5:
                connection["last_keepalive"] = asyncio.get_event_loop().time()
                asyncio.create_task(self.send_control_message(session_id, "KeepAlive"))
            
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed while sending audio for session: {session_id}")
            return False
        except Exception as e:
            logger.error(f"Error processing audio with Deepgram: {e}")
            return False
    
    async def send_control_message(self, session_id: str, message_type: str) -> bool:
        """
        Send a control message to Deepgram.
        
        Args:
            session_id: The session identifier
            message_type: The type of control message to send (Finalize, CloseStream, KeepAlive)
            
        Returns:
            bool: True if message was sent successfully
        """
        if session_id not in self.active_connections:
            logger.error(f"No active Deepgram connection for session: {session_id}")
            return False
        
        try:
            deepgram_ws = self.active_connections[session_id]["websocket"]
            
            # Check if connection is open
            if not deepgram_ws.open:
                logger.warning(f"WebSocket connection closed for session: {session_id}")
                return False
            
            # Create control message according to Deepgram documentation
            control_message = {"type": message_type}
            
            # Send control message
            await deepgram_ws.send(json.dumps(control_message))
            logger.debug(f"Sent {message_type} control message for session: {session_id}")
            
            return True
                
        except Exception as e:
            logger.error(f"Error sending control message: {e}")
            return False
    
    async def finalize_transcription(self, session_id: str) -> bool:
        """
        Force Deepgram to finalize the current transcription.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if finalize request was sent successfully
        """
        return await self.send_control_message(session_id, "Finalize")
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close a transcription session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if session was closed successfully
        """
        if session_id not in self.active_connections:
            logger.warning(f"No active Deepgram connection to close for session: {session_id}")
            return True
        
        try:
            connection = self.active_connections[session_id]
            
            # Try to finalize transcription first
            try:
                await self.send_control_message(session_id, "Finalize")
                # Give a short time for final results to arrive
                await asyncio.sleep(0.5)
            except Exception as finalize_error:
                logger.warning(f"Error finalizing transcription: {finalize_error}")
            
            # Try to send close stream message
            try:
                await self.send_control_message(session_id, "CloseStream")
            except Exception as close_error:
                logger.warning(f"Error sending close stream message: {close_error}")
            
            # Cancel the task if running
            if connection["task"] and not connection["task"].done():
                connection["task"].cancel()
                try:
                    await connection["task"]
                except asyncio.CancelledError:
                    pass
            
            # Close the WebSocket
            if connection["websocket"].open:
                await connection["websocket"].close()
            
            # Remove the connection
            self.active_connections.pop(session_id, None)
            
            logger.info(f"Deepgram transcription session closed: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing Deepgram transcription session: {e}")
            return False


# Global Deepgram STT service instance
deepgram_stt_service = DeepgramSTTService()