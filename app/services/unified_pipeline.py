"""
Unified Call Pipeline System.
Provides a cohesive flow that integrates STT, LLM, and TTS services into a single pipeline.
"""

import logging
import asyncio
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field

from app.services.deepgram_stt_service import deepgram_stt_service
from app.services.llm_service import openai_llm_service
# from app.services.rag_llm_service import get_rag_llm_service  # Not using RAG for now
from app.services.tts_service import elevenlabs_tts_service
from app.services.async_call_service import async_call_service
from app.db.crud import get_assistant, tool_crud
from app.config.settings import settings

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline state enumeration."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    stt_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    total_latency: float = 0.0
    error_count: int = 0
    turn_count: int = 0
    start_time: float = field(default_factory=time.monotonic)


@dataclass
class PipelineConfig:
    """Pipeline configuration - all values come from database."""
    # STT configuration
    stt_model: str
    stt_language: str
    stt_provider: str
    
    # LLM configuration
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_system_prompt: str
    
    # TTS configuration
    tts_voice_id: str
    tts_model: str
    tts_stability: float
    tts_similarity_boost: float
    tts_output_format: str
    
    # Pipeline settings
    max_conversation_history: int
    interim_timeout_seconds: float
    
    # RAG (Knowledge Base) settings
    enable_rag: bool
    rag_max_results: int
    rag_score_threshold: float
    rag_max_context_length: int


class UnifiedCallPipeline:
    """
    Unified pipeline that coordinates STT, LLM, and TTS services.
    
    This class provides a single entry point for audio processing that handles
    the complete flow: Audio → STT → LLM → TTS → Audio
    """
    
    def __init__(self, call_sid: str, assistant_id: uuid.UUID, audio_callback: Callable[[str], Coroutine], text_callback: Optional[Callable[[str, str, bool], Coroutine]] = None):
        """
        Initialize the unified pipeline.
        
        Args:
            call_sid: Unique call identifier
            assistant_id: Assistant configuration ID
            audio_callback: Callback function for sending audio to Twilio
            text_callback: Optional callback function for sending text messages (for web clients)
        """
        self.call_sid = call_sid
        self.assistant_id = assistant_id
        self.audio_callback = audio_callback
        self.text_callback = text_callback
        
        # Detect if this is a web-based stream (chat/webrtc) vs phone call
        self.is_web_stream = call_sid.startswith("chat_") or call_sid.startswith("webrtc_")
        
        # Pipeline state
        self.state = PipelineState.IDLE
        self.config = None  # Will be set from database
        self.metrics = PipelineMetrics()
        
        # Service sessions
        self.stt_session_id = f"stt_{call_sid}"
        self.tts_session_id = f"tts_{call_sid}"
        
        # Conversation state
        self.conversation_history = []
        self.current_transcript = ""
        self.current_response = ""
        self.current_confidence = 1.0  # Default confidence for STT
        
        # Interim transcript timeout handling
        self.last_interim_time = None
        self.interim_timeout_task = None
        
        # Pipeline control
        self._shutdown_event = asyncio.Event()
        self._processing_lock = asyncio.Lock()
        # Removed barge-in functionality
        
        # Assistant configuration
        self.assistant_config = None
        self.assistant_tools = []  # Store assistant tools for tool calling
        
        logger.info(f"Unified pipeline initialized for call: {call_sid}")
    
    async def initialize(self) -> bool:
        """
        Initialize the pipeline with assistant configuration.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info(f"Initializing pipeline for call: {self.call_sid}")
            
            # Load assistant configuration
            from sqlalchemy.orm import Session
            from app.db.database import SessionLocal
            db_session = SessionLocal()
            
            assistant = get_assistant(db_session, self.assistant_id)
            if not assistant:
                logger.error(f"Assistant not found: {self.assistant_id}")
                return False
            
            self.assistant_config = assistant.to_dict()
            
            # Load assistant tools for tool calling
            try:
                self.assistant_tools = tool_crud.get_assistant_tools(
                    db_session, self.assistant_id, assistant.organization_id, enabled_only=True
                )
                logger.info(f"Loaded {len(self.assistant_tools)} tools for assistant {self.assistant_id}")
            except Exception as e:
                logger.warning(f"Failed to load assistant tools: {e}")
                self.assistant_tools = []
            
            db_session.close()
            
            # Configure pipeline based on assistant settings
            await self._configure_pipeline()
            
            # Initialize services
            await self._initialize_services()
            
            # Using regular LLM service (no RAG initialization needed)
            
            # Start in IDLE state, but we'll transition to continuous listening mode
            self.state = PipelineState.IDLE
            logger.info(f"Pipeline initialized successfully for call: {self.call_sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.state = PipelineState.ERROR
            return False
    
    async def _configure_pipeline(self):
        """Configure pipeline based on assistant settings."""
        if not self.assistant_config:
            raise ValueError("Assistant configuration is required")
        
        # Configure STT
        transcriber_config = self.assistant_config.get("transcriber", {})
        if not transcriber_config.get("model"):
            raise ValueError("STT model is required")
        if not transcriber_config.get("language"):
            raise ValueError("STT language is required")
        if not transcriber_config.get("provider"):
            raise ValueError("STT provider is required")
        
        # Configure LLM
        model_config = self.assistant_config.get("model", {})
        if not model_config.get("model"):
            raise ValueError("LLM model is required")
        if not model_config.get("maxTokens"):
            raise ValueError("LLM max tokens is required")
        if model_config.get("temperature") is None:
            raise ValueError("LLM temperature is required")
        if not model_config.get("systemPrompt"):
            raise ValueError("LLM system prompt is required")
        
        # Configure TTS
        voice_config = self.assistant_config.get("voice", {})
        if not voice_config.get("voiceId"):
            raise ValueError("TTS voice ID is required")
        if not voice_config.get("model"):
            raise ValueError("TTS model is required")
        if voice_config.get("stability") is None:
            raise ValueError("TTS stability is required")
        if voice_config.get("similarityBoost") is None:
            raise ValueError("TTS similarity boost is required")
        # Output format comes from settings, not database
        output_format = settings.elevenlabs_output_format
        logger.info(f"🎤 TTS Config from DB - Voice: {voice_config['voiceId']}, Model: {voice_config['model']}, Stability: {voice_config['stability']}, Similarity: {voice_config['similarityBoost']}")
        logger.info(f"🎤 TTS Output format from settings: {output_format}")
        
        # Configure RAG
        rag_config = self.assistant_config.get("rag", {})
        if rag_config.get("enabled") is None:
            raise ValueError("RAG enabled setting is required")
        
        # Create configuration from database values
        self.config = PipelineConfig(
            # STT configuration
            stt_model=transcriber_config["model"],
            stt_language=transcriber_config["language"],
            stt_provider=transcriber_config["provider"].lower(),
            
            # LLM configuration
            llm_model=model_config["model"],
            llm_max_tokens=model_config["maxTokens"],
            llm_temperature=model_config["temperature"],
            llm_system_prompt=model_config["systemPrompt"],
            
            # TTS configuration
            tts_voice_id=voice_config["voiceId"],
            tts_model=voice_config["model"],
            tts_stability=voice_config["stability"],
            tts_similarity_boost=voice_config["similarityBoost"],
            tts_output_format=output_format,
            
            # Pipeline settings
            max_conversation_history=10,  # Default value
            interim_timeout_seconds=2.0,  # Default value
            
            # RAG configuration
            enable_rag=rag_config["enabled"],
            rag_max_results=rag_config.get("maxResults", 3),
            rag_score_threshold=rag_config.get("scoreThreshold", 0.7),
            rag_max_context_length=rag_config.get("maxContextLength", 2000)
        )
    
    async def _initialize_services(self):
        """Initialize STT and TTS services."""
        # Initialize STT service
        if self.config.stt_provider == "deepgram":
            success = await deepgram_stt_service.create_transcription_session(
                self.stt_session_id,
                self._handle_stt_delta,
                self._handle_stt_complete,
                model=self.config.stt_model,
                language=self.config.stt_language,
                on_speech_started=self._handle_speech_started,
                is_web_stream=self.is_web_stream
            )
            if not success:
                raise Exception("Failed to initialize Deepgram STT service")
        
        # Initialize TTS service
        logger.info(f"🎤 Creating TTS session with config - Voice: {self.config.tts_voice_id}, Model: {self.config.tts_model}, Stability: {self.config.tts_stability}, Similarity: {self.config.tts_similarity_boost}, Format: {self.config.tts_output_format}")
        await elevenlabs_tts_service.create_tts_session(
            self.tts_session_id,
            self._handle_tts_audio,
            voice_id=self.config.tts_voice_id,
            model_id=self.config.tts_model,
            stability=self.config.tts_stability,
            similarity_boost=self.config.tts_similarity_boost,
            output_format=self.config.tts_output_format,
            on_completion=self._handle_tts_completion
        )
        
        logger.info("Services initialized successfully")
    
    async def process_audio(self, audio_data: str) -> bool:
        """
        Process incoming audio through the unified pipeline.
        
        This method ALWAYS processes audio to enable continuous listening and barge-in.
        
        Args:
            audio_data: Base64-encoded audio data
            
        Returns:
            bool: True if audio was processed successfully
        """
        if self.state == PipelineState.SHUTDOWN:
            logger.warning(f"Pipeline is shutting down, ignoring audio for call: {self.call_sid}")
            return False
        
        try:
            # ALWAYS process audio for continuous listening and barge-in
            # The STT service will handle speech detection and only trigger callbacks when speech is detected
            
            # Check if pipeline is properly configured
            if not self.config:
                logger.error("Pipeline not properly configured - config is None")
                return False
            
            # Send audio to STT service regardless of current state
            if self.config.stt_provider == "deepgram":
                await deepgram_stt_service.process_audio(self.stt_session_id, audio_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing audio in pipeline: {e}")
            await self._handle_pipeline_error("audio_processing", e)
            return False
    
    async def _handle_stt_delta(self, session_id: str, text: str, confidence: float = 0.0):
        """Handle STT transcription delta."""
        if session_id != self.stt_session_id:
            return
        
        # Store the current confidence for final transcript
        self.current_confidence = confidence
        # Reduced debug logging for performance
        self.current_transcript = text
        
        # Send streaming user text to frontend if text callback is available
        if self.text_callback and self.is_web_stream and text.strip():
            await self.text_callback("user_streaming", text, is_final=False)
        
        # Track interim transcript timing for timeout handling
        self.last_interim_time = time.monotonic()
        
        # Cancel any existing timeout task
        if self.interim_timeout_task and not self.interim_timeout_task.done():
            self.interim_timeout_task.cancel()
            logger.debug(f"⏹️ Cancelled previous interim timeout for call {self.call_sid}")
        
        # Start new timeout task
        self.interim_timeout_task = asyncio.create_task(
            self._handle_interim_timeout()
        )
        
        logger.debug(f"🔄 Started interim timeout timer for call {self.call_sid}: '{text[:30]}...' (timeout: {self.config.interim_timeout_seconds}s)")
    
    async def _handle_interim_timeout(self):
        """Handle timeout for interim transcripts."""
        try:
            # Wait for the timeout period
            await asyncio.sleep(self.config.interim_timeout_seconds)
            
            # Check if we still have an interim transcript that hasn't been finalized
            if (self.current_transcript and 
                self.last_interim_time and 
                time.monotonic() - self.last_interim_time >= self.config.interim_timeout_seconds):
                
                logger.info(f"⏰ Interim transcript timeout for call {self.call_sid}, treating as final: '{self.current_transcript}'")
                
                # Treat the interim transcript as final
                await self._handle_stt_complete(self.stt_session_id, self.current_transcript)
                
        except asyncio.CancelledError:
            # Task was cancelled, which is expected when new interim data arrives
            pass
        except Exception as e:
            logger.error(f"Error in interim timeout handler: {e}")

    async def _handle_stt_complete(self, session_id: str, text: str):
        """Handle STT transcription completion."""
        if session_id != self.stt_session_id:
            return
        
        # Cancel any pending timeout task since we have a final transcript
        if self.interim_timeout_task and not self.interim_timeout_task.done():
            self.interim_timeout_task.cancel()
            logger.debug(f"✅ Cancelled interim timeout for call {self.call_sid} - final transcript received")
            try:
                await self.interim_timeout_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"STT complete for call {self.call_sid}: {text}")
        
        # Mark STT completion time
        stt_end_time = time.monotonic()
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": text
        })
        
        # Store user transcript message asynchronously with actual confidence score (fire-and-forget)
        confidence_score = getattr(self, 'current_confidence', 1.0)
        logger.info(f"🔍 Queuing user transcript for call {self.call_sid}: '{text}' (confidence: {confidence_score:.2f})")
        # Fire-and-forget: Don't await to avoid blocking the pipeline
        asyncio.create_task(async_call_service.add_transcript_message(
            twilio_call_sid=self.call_sid,
            speaker="user",
            message=text,
            confidence=confidence_score
        ))
        
        # Send final user message to frontend if text callback is available
        if self.text_callback and self.is_web_stream:
            await self.text_callback("user_streaming", text, is_final=True)
        
        # Limit conversation history
        if len(self.conversation_history) > self.config.max_conversation_history * 2:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_history * 2:]
        
        # Process the complete transcript
        
        # Transition to processing state and generate response
        logger.info(f"🔄 Processing user input for call {self.call_sid}: '{text}' (current state: {self.state})")
        # Reduced debug logging for performance
        self.state = PipelineState.PROCESSING
        
        # Record STT completion time for latency tracking
        # STT latency is from when the user started speaking
        if hasattr(self, '_turn_start_time'):
            self.metrics.stt_latency = (stt_end_time - self._turn_start_time) * 1000
        else:
            self.metrics.stt_latency = 0.0
        logger.info(f"⏱️ STT completed in {self.metrics.stt_latency:.1f}ms for call {self.call_sid}")
        
        await self._generate_llm_response(text, stt_end_time)
    
    async def _handle_speech_started(self, session_id: str):
        """Handle speech started event."""
        if session_id != self.stt_session_id:
            return
        
        # Reduced debug logging for performance
        self.metrics.start_time = time.monotonic()
        self._turn_start_time = time.monotonic()  # Record when user started speaking
        self._tts_first_audio_recorded = False  # Reset TTS flag for new turn
        
        # Note: We don't trigger barge-in here because SpeechStarted can be triggered by TTS audio
        # Barge-in should only be triggered when we have actual user transcript content
    
    async def _generate_llm_response(self, transcript: str, stt_end_time: float):
        """Generate LLM response and trigger TTS."""
        try:
            async with self._processing_lock:
                if self.state != PipelineState.PROCESSING:
                    return
                
                logger.info(f"Generating LLM response for call {self.call_sid}")
                llm_start_time = time.monotonic()
                
                # Prepare system prompt
                system_prompt = self._get_system_prompt()
                
                # Get conversation history (last 10 messages)
                recent_history = self.conversation_history[-self.config.max_conversation_history:]
                
                # Define content delta handler for streaming with batching
                tts_buffer = []
                buffer_size = 0
                max_buffer_size = 100  # Batch up to 100 characters for better performance
                
                async def handle_content_delta(delta: str):
                    nonlocal tts_buffer, buffer_size
                    if self.state == PipelineState.PROCESSING:
                        self.current_response += delta
                        tts_buffer.append(delta)
                        buffer_size += len(delta)
                        
                        # Send streaming assistant text to frontend
                        if self.text_callback and self.is_web_stream:
                            await self.text_callback("assistant_streaming", self.current_response, is_final=False)
                        
                        # Send batched text when buffer is full or contains sentence-ending punctuation
                        if (buffer_size >= max_buffer_size or 
                            any(punct in delta for punct in ['.', '!', '?', '\n'])):
                            batched_text = ''.join(tts_buffer)
                            logger.debug(f"Sending batched TTS for call {self.call_sid}: '{batched_text[:30]}...'")
                            await elevenlabs_tts_service.speak_text(
                                self.tts_session_id, 
                                batched_text, 
                                is_final=False
                            )
                            tts_buffer = []
                            buffer_size = 0
                
                # Generate response with LLM service (with tool calling if tools are available)
                logger.info(f"Using LLM model: {self.config.llm_model}")
                
                if self.assistant_tools:
                    logger.info(f"Using tool-enabled LLM with {len(self.assistant_tools)} tools")
                    response = await openai_llm_service.generate_response_with_tools(
                        text=transcript,
                        on_content_delta=handle_content_delta,
                        assistant_tools=self.assistant_tools,
                        conversation_history=recent_history,
                        custom_system_prompt=system_prompt,
                        model=self.config.llm_model,
                        max_tokens=self.config.llm_max_tokens,
                        temperature=self.config.llm_temperature
                    )
                else:
                    logger.info("Using regular LLM (no tools available)")
                    response = await openai_llm_service.generate_response(
                        text=transcript,
                        on_content_delta=handle_content_delta,
                        conversation_history=recent_history,
                        custom_system_prompt=system_prompt,
                        model=self.config.llm_model,
                        max_tokens=self.config.llm_max_tokens,
                        temperature=self.config.llm_temperature
                    )
                
                # Mark LLM completion time
                llm_end_time = time.monotonic()
                self.metrics.llm_latency = (llm_end_time - llm_start_time) * 1000
                self._llm_end_time = llm_end_time  # Store for TTS latency calculation
                logger.info(f"⏱️ LLM completed in {self.metrics.llm_latency:.1f}ms for call {self.call_sid}")
                
                # Send any remaining buffered text
                if tts_buffer:
                    remaining_text = ''.join(tts_buffer)
                    logger.debug(f"Sending remaining TTS buffer for call {self.call_sid}: '{remaining_text[:30]}...'")
                    await elevenlabs_tts_service.speak_text(
                        self.tts_session_id, 
                        remaining_text, 
                        is_final=False
                    )
                
                # Finalize TTS
                logger.debug(f"Finalizing TTS for call {self.call_sid}")
                await elevenlabs_tts_service.speak_text(self.tts_session_id, "", is_final=True)
                
                # Add assistant response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Store assistant transcript message asynchronously (system confidence = 1.0) (fire-and-forget)
                logger.info(f"🔍 Queuing assistant transcript for call {self.call_sid}: '{response[:50]}...'")
                # Fire-and-forget: Don't await to avoid blocking the pipeline
                asyncio.create_task(async_call_service.add_transcript_message(
                    twilio_call_sid=self.call_sid,
                    speaker="assistant",
                    message=response,
                    confidence=1.0  # System messages always have confidence = 1.0
                ))
                
                # Send final assistant message to frontend if text callback is available
                if self.text_callback and self.is_web_stream:
                    await self.text_callback("assistant_streaming", response, is_final=True)
                
                # Update metrics
                self.metrics.turn_count += 1
                self.metrics.total_latency = (llm_end_time - stt_end_time) * 1000
                logger.info(f"⏱️ TOTAL TURN LATENCY: {self.metrics.total_latency:.1f}ms for call {self.call_sid} (STT: {self.metrics.stt_latency:.1f}ms, LLM: {self.metrics.llm_latency:.1f}ms, TTS: {self.metrics.tts_latency:.1f}ms)")
                
                # Transition to speaking state
                logger.info(f"🎤 AI starting to speak for call {self.call_sid}: '{response[:50]}...'")
                # Reduced debug logging for performance
                self.state = PipelineState.SPEAKING
                
                logger.info(f"LLM response generated for call {self.call_sid}: {response[:50]}...")
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            await self._handle_pipeline_error("llm_generation", e)
    
    async def _handle_tts_audio(self, session_id: str, audio_data: str):
        """Handle TTS audio output."""
        if session_id != self.tts_session_id:
            return
        
        # Record TTS first audio timing
        if not hasattr(self, '_tts_first_audio_recorded'):
            tts_first_audio_time = time.monotonic()
            if hasattr(self, '_llm_end_time'):
                self.metrics.tts_latency = (tts_first_audio_time - self._llm_end_time) * 1000
            else:
                self.metrics.tts_latency = 0.0
            self._tts_first_audio_recorded = True
            logger.info(f"⏱️ TTS first audio in {self.metrics.tts_latency:.1f}ms for call {self.call_sid}")
        
        # Send audio to client (Twilio or WebSocket)
        await self.audio_callback(audio_data)
    
    async def _handle_tts_completion(self, session_id: str):
        """Handle TTS completion."""
        if session_id != self.tts_session_id:
            return
        
        logger.debug(f"TTS completed for call {self.call_sid}")
        # Note: We don't transition back to LISTENING here because we want continuous listening
        # The pipeline should always be listening for user input, even while speaking
        # This enables proper barge-in functionality
    
    # Removed barge-in functionality
    
    def _get_system_prompt(self) -> str:
        """Get system prompt from assistant configuration."""
        if not self.assistant_config:
            return "You are a helpful voice assistant. Be concise and conversational."
        
        model_config = self.assistant_config.get("model", {})
        messages = model_config.get("messages", [])
        
        for message in messages:
            if message.get("role") == "system":
                return message.get("content", "")
        
        return "You are a helpful voice assistant. Be concise and conversational."
    
    async def play_greeting(self):
        """Play the initial greeting to the caller."""
        if not self.assistant_config:
            greeting_text = "Hello! I'm your AI assistant. How can I help you today?"
        else:
            greeting_text = self.assistant_config.get("firstMessage", "Hello! How can I help you today?")
        
        logger.info(f"Playing greeting for call {self.call_sid}: {greeting_text}")
        
        # Set state to speaking
        self.state = PipelineState.SPEAKING
        
        # Play greeting
        greeting_start_time = time.monotonic()
        await elevenlabs_tts_service.speak_text(self.tts_session_id, greeting_text, is_final=False)
        await elevenlabs_tts_service.speak_text(self.tts_session_id, "", is_final=True)
        greeting_latency = (time.monotonic() - greeting_start_time) * 1000
        
        logger.info(f"⏱️ Greeting TTS latency: {greeting_latency:.1f}ms for call {self.call_sid}")
        
        # Add greeting to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": greeting_text
        })
        
        # Send greeting to frontend if text callback is available
        if self.text_callback and self.is_web_stream:
            await self.text_callback("assistant", greeting_text)
        
        # Note: We don't transition to LISTENING state here because we want continuous listening
        # The pipeline should always be listening for user input, even while speaking
        # This enables proper barge-in functionality
        
        logger.info(f"Greeting completed for call {self.call_sid} - pipeline now in continuous listening mode")
    
    async def _handle_pipeline_error(self, error_type: str, error: Exception):
        """Handle pipeline errors with recovery."""
        logger.error(f"Pipeline error ({error_type}) for call {self.call_sid}: {error}")
        
        self.metrics.error_count += 1
        self.state = PipelineState.ERROR
        
        # Attempt recovery based on error type
        if error_type == "audio_processing":
            # Audio processing errors are usually recoverable
            # Note: We don't change state - pipeline continues listening
            logger.info(f"Audio processing error recovered for call {self.call_sid}")
        elif error_type == "llm_generation":
            # LLM errors might be recoverable
            # Note: We don't change state - pipeline continues listening
            logger.info(f"LLM generation error recovered for call {self.call_sid}")
        else:
            # Other errors might require pipeline restart
            logger.error(f"Unrecoverable pipeline error for call {self.call_sid}")
    
    async def shutdown(self):
        """Shutdown the pipeline and cleanup all resources including audio and memory."""
        logger.info(f"Shutting down pipeline for call {self.call_sid}")
        
        self.state = PipelineState.SHUTDOWN
        self._shutdown_event.set()
        
        try:
            # Cancel any pending timeout task
            if self.interim_timeout_task and not self.interim_timeout_task.done():
                self.interim_timeout_task.cancel()
                try:
                    await self.interim_timeout_task
                except asyncio.CancelledError:
                    pass
            
            # Close STT session (clears audio buffers and connections)
            if self.config and self.config.stt_provider == "deepgram":
                await deepgram_stt_service.close_session(self.stt_session_id)
            
            # Close TTS session (clears audio buffers and connections)
            await elevenlabs_tts_service.close_session(self.tts_session_id)
            
            # Clear conversation history and audio data from memory
            self.conversation_history.clear()
            self.current_transcript = ""
            self.interim_transcript = ""
            
            # Clear any cached audio data
            if hasattr(self, 'cached_audio_data'):
                self.cached_audio_data.clear()
            
            # Reset metrics
            self.metrics = PipelineMetrics()
            
            logger.info(f"Pipeline shutdown completed with full cleanup for call {self.call_sid}")
            
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        return {
            "call_sid": self.call_sid,
            "state": self.state.value,
            "turn_count": self.metrics.turn_count,
            "error_count": self.metrics.error_count,
            "stt_latency_ms": self.metrics.stt_latency,
            "llm_latency_ms": self.metrics.llm_latency,
            "tts_latency_ms": self.metrics.tts_latency,
            "total_latency_ms": self.metrics.total_latency,
            "conversation_length": len(self.conversation_history)
        }


class UnifiedPipelineManager:
    """Manager for multiple unified pipelines."""
    
    def __init__(self):
        """Initialize the pipeline manager."""
        self.active_pipelines: Dict[str, UnifiedCallPipeline] = {}
        self._initialization_lock = asyncio.Lock()
        self._services_initialized = False
    
    async def initialize_services(self):
        """Initialize all services."""
        async with self._initialization_lock:
            if self._services_initialized:
                return True
            
            try:
                logger.info("Initializing unified pipeline services...")
                
                # Initialize individual services
                results = await asyncio.gather(
                    deepgram_stt_service.initialize(),
                    openai_llm_service.initialize(),
                    elevenlabs_tts_service.initialize(),
                    return_exceptions=True
                )
                
                # Check for exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        service_names = ["Deepgram STT", "OpenAI LLM", "ElevenLabs TTS"]
                        logger.error(f"Failed to initialize {service_names[i]} service: {result}")
                        return False
                
                self._services_initialized = all(results)
                
                if self._services_initialized:
                    logger.info("All unified pipeline services initialized successfully")
                else:
                    logger.error("Failed to initialize all unified pipeline services")
                
                return self._services_initialized
                
            except Exception as e:
                logger.error(f"Error initializing unified pipeline services: {e}")
                return False
    
    async def create_pipeline(self, call_sid: str, assistant_id: uuid.UUID, 
                            audio_callback: Callable[[str], Coroutine],
                            text_callback: Optional[Callable[[str, str, bool], Coroutine]] = None,
                            config_overrides: Optional[Dict[str, Any]] = None) -> UnifiedCallPipeline:
        """
        Create a new unified pipeline for a call.
        
        Args:
            call_sid: Unique call identifier
            assistant_id: Assistant configuration ID
            audio_callback: Callback function for sending audio to client
            text_callback: Optional callback function for sending text messages
            config_overrides: Optional configuration overrides
            
        Returns:
            UnifiedCallPipeline: The created pipeline instance
        """
        if not self._services_initialized:
            await self.initialize_services()
        
        pipeline = UnifiedCallPipeline(call_sid, assistant_id, audio_callback, text_callback)
        
        # Apply config overrides if provided
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(pipeline.config, key):
                    setattr(pipeline.config, key, value)
                    logger.info(f"Applied config override: {key} = {value}")
        
        success = await pipeline.initialize()
        if not success:
            logger.error(f"Failed to initialize pipeline for call: {call_sid}")
            return None
        
        self.active_pipelines[call_sid] = pipeline
        logger.info(f"Created unified pipeline for call: {call_sid}")
        
        return pipeline
    
    async def get_pipeline(self, call_sid: str) -> Optional[UnifiedCallPipeline]:
        """Get an active pipeline by call SID."""
        return self.active_pipelines.get(call_sid)
    
    async def remove_pipeline(self, call_sid: str):
        """Remove and shutdown a pipeline."""
        pipeline = self.active_pipelines.pop(call_sid, None)
        if pipeline:
            await pipeline.shutdown()
            logger.info(f"Removed unified pipeline for call: {call_sid}")


# Global unified pipeline manager instance
unified_pipeline_manager = UnifiedPipelineManager()
