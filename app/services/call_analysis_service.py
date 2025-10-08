"""
Call Analysis Service.

Provides AI-powered analysis of call transcripts to determine:
- Call success (whether the call achieved its goal)
- Call summary (concise overview of the conversation)
- Sentiment analysis (overall sentiment score from 0 to 1)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import openai
from app.config.settings import settings
from app.services.llm_service import openai_llm_service

logger = logging.getLogger(__name__)


@dataclass
class CallAnalysisResult:
    """Result of call analysis."""
    call_success: bool
    call_summary: str
    sentiment_score: float  # 0.0 to 1.0
    analysis_metadata: Dict[str, Any]
    detailed_analysis: Dict[str, Any]  # Additional analysis data for UI


class CallAnalysisService:
    """
    Service for analyzing call transcripts using OpenAI.
    
    This service provides comprehensive analysis of call conversations including
    success detection, summarization, and sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the call analysis service."""
        self.client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the OpenAI client.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if not settings.OPENAI_API_KEY:
                logger.error("OpenAI API key not configured")
                return False
            
            # Use the existing LLM service client
            if not openai_llm_service.client:
                await openai_llm_service.initialize()
            
            self.client = openai_llm_service.client
            self._initialized = True
            logger.info("Call analysis service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize call analysis service: {e}")
            return False
    
    async def analyze_call(
        self, 
        transcript_data: List[Dict[str, Any]], 
        system_prompt: str,
        assistant_id: Optional[str] = None
    ) -> CallAnalysisResult:
        """
        Analyze a call transcript to determine success, generate summary, and calculate sentiment.
        
        Args:
            transcript_data: List of transcript messages with speaker, message, timestamp, confidence
            system_prompt: The system prompt used for the assistant during the call
            assistant_id: Optional assistant ID for context
            
        Returns:
            CallAnalysisResult: Analysis results including success, summary, and sentiment
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.client:
            logger.error("Call analysis service not properly initialized")
            raise Exception("Call analysis service not initialized")
        
        try:
            # Check if we have transcript data to analyze
            if not transcript_data or len(transcript_data) == 0:
                logger.warning("No transcript data available for analysis")
                return self._get_default_analysis_result()
            
            # Convert transcript to conversation format
            conversation = self._format_transcript_for_analysis(transcript_data)
            
            # Check if conversation is empty after formatting
            if not conversation.strip():
                logger.warning("Empty conversation after formatting transcript data")
                return self._get_default_analysis_result()
            
            # Perform all analyses in parallel for efficiency
            success_task = asyncio.create_task(self._analyze_call_success(conversation, system_prompt))
            summary_task = asyncio.create_task(self._generate_call_summary(conversation, system_prompt))
            sentiment_task = asyncio.create_task(self._analyze_sentiment(conversation))
            detailed_task = asyncio.create_task(self._generate_detailed_analysis(conversation, system_prompt))
            
            # Wait for all analyses to complete
            call_success, call_summary, sentiment_score, detailed_analysis = await asyncio.gather(
                success_task, summary_task, sentiment_task, detailed_task
            )
            
            # Create analysis metadata
            analysis_metadata = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "transcript_length": len(transcript_data),
                "conversation_turns": len([msg for msg in transcript_data if msg.get("speaker") == "user"]),
                "assistant_id": assistant_id,
                "analysis_version": "1.0"
            }
            
            result = CallAnalysisResult(
                call_success=call_success,
                call_summary=call_summary,
                sentiment_score=sentiment_score,
                analysis_metadata=analysis_metadata,
                detailed_analysis=detailed_analysis
            )
            
            logger.info(f"Call analysis completed - Success: {call_success}, Sentiment: {sentiment_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing call: {e}")
            # Return default values on error
            return CallAnalysisResult(
                call_success=False,
                call_summary="Analysis failed due to error",
                sentiment_score=0.5,  # Neutral sentiment
                analysis_metadata={"error": str(e), "analyzed_at": datetime.utcnow().isoformat()},
                detailed_analysis={"error": "Analysis failed", "sentiment_breakdown": {}, "conversation_metrics": {}}
            )
    
    def _format_transcript_for_analysis(self, transcript_data: List[Dict[str, Any]]) -> str:
        """
        Format transcript data into a readable conversation format.
        
        Args:
            transcript_data: Raw transcript data from database
            
        Returns:
            str: Formatted conversation text
        """
        conversation_parts = []
        
        for message in transcript_data:
            speaker = message.get("speaker", "unknown")
            content = message.get("message", "")
            timestamp = message.get("timestamp", "")
            
            # Format timestamp for readability
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            else:
                time_str = ""
            
            # Add speaker label and content
            if speaker == "user":
                conversation_parts.append(f"[{time_str}] User: {content}")
            elif speaker == "assistant":
                conversation_parts.append(f"[{time_str}] Assistant: {content}")
            else:
                conversation_parts.append(f"[{time_str}] {speaker.title()}: {content}")
        
        return "\n".join(conversation_parts)
    
    async def _analyze_call_success(self, conversation: str, system_prompt: str) -> bool:
        """
        Analyze whether the call was successful based on the conversation and system prompt.
        
        Args:
            conversation: Formatted conversation text
            system_prompt: The system prompt that defined the assistant's purpose
            
        Returns:
            bool: True if the call was successful, False otherwise
        """
        try:
            prompt = f"""
You are analyzing a customer service call to determine if it was successful.

System Prompt (Assistant's Purpose):
{system_prompt}

Call Transcript:
{conversation}

Based on the system prompt and the conversation, determine if this call was successful. Consider:
1. Did the assistant fulfill its intended purpose as defined in the system prompt?
2. Was the customer's issue resolved or their request fulfilled?
3. Did the conversation end positively?
4. Were there any clear indicators of success (ticket created, problem solved, information provided, etc.)?

Respond with ONLY "SUCCESS" or "FAILURE" - no other text.
"""
            
            response = await self._make_openai_request(prompt, max_tokens=10)
            result = response.strip().upper()
            
            return result == "SUCCESS"
            
        except Exception as e:
            logger.error(f"Error analyzing call success: {e}")
            return False
    
    async def _generate_call_summary(self, conversation: str, system_prompt: str) -> str:
        """
        Generate a concise summary of the call.
        
        Args:
            conversation: Formatted conversation text
            system_prompt: The system prompt that defined the assistant's purpose
            
        Returns:
            str: Concise summary of the call
        """
        try:
            prompt = f"""
You are summarizing a customer service call.

System Prompt (Assistant's Purpose):
{system_prompt}

Call Transcript:
{conversation}

Generate a concise summary (2-3 sentences) that captures:
1. What the customer wanted/needed
2. How the assistant helped
3. The outcome of the call

Keep it professional and factual. Do not include timestamps or speaker labels.
"""
            
            response = await self._make_openai_request(prompt, max_tokens=150)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating call summary: {e}")
            return "Summary generation failed due to error."
    
    async def _analyze_sentiment(self, conversation: str) -> float:
        """
        Analyze the overall sentiment of the call.
        
        Args:
            conversation: Formatted conversation text
            
        Returns:
            float: Sentiment score from 0.0 (negative) to 1.0 (positive)
        """
        try:
            prompt = f"""
You are analyzing the sentiment of a customer service call.

Call Transcript:
{conversation}

Analyze the overall sentiment of this conversation. Consider:
1. Customer's emotional state throughout the call
2. Tone of the conversation
3. Satisfaction indicators
4. Overall mood and outcome

Respond with ONLY a number between 0.0 and 1.0 where:
- 0.0 = Very negative (angry, frustrated, disappointed)
- 0.5 = Neutral (neither positive nor negative)
- 1.0 = Very positive (happy, satisfied, pleased)

Examples:
- 0.2 = Customer was frustrated and issue not resolved
- 0.5 = Neutral conversation, standard interaction
- 0.8 = Customer was satisfied with the service
- 1.0 = Customer was very happy and grateful

Respond with ONLY the number, no other text.
"""
            
            response = await self._make_openai_request(prompt, max_tokens=10)
            
            try:
                score = float(response.strip())
                # Ensure score is within valid range
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Invalid sentiment score received: {response}")
                return 0.5  # Default to neutral
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.5  # Default to neutral
    
    async def _generate_detailed_analysis(self, conversation: str, system_prompt: str) -> Dict[str, Any]:
        """
        Generate detailed analysis data for the UI including sentiment breakdown and conversation metrics.
        
        Args:
            conversation: Formatted conversation text
            system_prompt: The system prompt that defined the assistant's purpose
            
        Returns:
            Dict[str, Any]: Detailed analysis data
        """
        try:
            prompt = f"""
You are analyzing a customer service call to provide detailed insights.

System Prompt (Assistant's Purpose):
{system_prompt}

Call Transcript:
{conversation}

Provide a detailed analysis in JSON format with the following structure:
{{
    "sentiment_breakdown": {{
        "customer_sentiment": "positive/negative/neutral",
        "assistant_performance": "excellent/good/average/poor",
        "overall_tone": "professional/friendly/formal/casual",
        "emotional_indicators": ["frustrated", "satisfied", "confused", "grateful", etc.]
    }},
    "conversation_metrics": {{
        "resolution_quality": "resolved/partially_resolved/not_resolved",
        "response_time": "fast/adequate/slow",
        "communication_effectiveness": "clear/unclear/mixed",
        "customer_satisfaction_indicators": ["thankful", "complaining", "asking_follow_up", etc.]
    }},
    "key_topics": ["topic1", "topic2", "topic3"],
    "action_items": ["item1", "item2", "item3"],
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}}

Respond with ONLY the JSON object, no other text.
"""
            
            response = await self._make_openai_request(prompt, max_tokens=500)
            
            try:
                import json
                detailed_data = json.loads(response.strip())
                return detailed_data
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON response for detailed analysis: {response}")
                return self._get_default_detailed_analysis()
                
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {e}")
            return self._get_default_detailed_analysis()
    
    def _get_default_detailed_analysis(self) -> Dict[str, Any]:
        """Get default detailed analysis data when analysis fails."""
        return {
            "sentiment_breakdown": {
                "customer_sentiment": "neutral",
                "assistant_performance": "average",
                "overall_tone": "professional",
                "emotional_indicators": []
            },
            "conversation_metrics": {
                "resolution_quality": "unknown",
                "response_time": "unknown",
                "communication_effectiveness": "unknown",
                "customer_satisfaction_indicators": []
            },
            "key_topics": [],
            "action_items": [],
            "improvement_suggestions": []
        }
    
    async def _make_openai_request(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Make a request to OpenAI API.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            
        Returns:
            str: The response text
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster, cheaper model for analysis
                messages=[
                    {"role": "system", "content": "You are an expert call analyst. Provide accurate, concise analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent analysis
                timeout=30
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
    
    def _get_default_analysis_result(self) -> CallAnalysisResult:
        """
        Get default analysis result for calls with no transcript data.
        
        Returns:
            CallAnalysisResult: Default analysis result
        """
        return CallAnalysisResult(
            call_success=False,
            call_summary="No transcript data available for analysis. This call may have been too short or had technical issues.",
            sentiment_score=0.5,  # Neutral sentiment
            analysis_metadata={
                "analyzed_at": datetime.utcnow().isoformat(),
                "transcript_length": 0,
                "conversation_turns": 0,
                "analysis_version": "1.0",
                "note": "Default analysis due to missing transcript data"
            },
            detailed_analysis={
                "sentiment_breakdown": {
                    "customer_sentiment": "Unknown - No transcript data",
                    "assistant_performance": "Unknown - No transcript data", 
                    "overall_tone": "Unknown - No transcript data",
                    "emotional_indicators": []
                },
                "conversation_metrics": {
                    "resolution_quality": "Unknown - No transcript data",
                    "response_time": "Unknown - No transcript data",
                    "communication_effectiveness": "Unknown - No transcript data",
                    "customer_satisfaction_indicators": []
                },
                "key_topics": [],
                "action_items": ["Investigate why no transcript was captured"],
                "improvement_suggestions": ["Ensure proper transcript capture for future calls"]
            }
        )


# Global instance
call_analysis_service = CallAnalysisService()
