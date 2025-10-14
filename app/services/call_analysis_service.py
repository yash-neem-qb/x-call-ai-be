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
            if not settings.openai_api_key:
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
                return self._get_default_comprehensive_analysis_result()
            
            # Convert transcript to conversation format
            conversation = self._format_transcript_for_analysis(transcript_data)
            
            # Check if conversation is empty after formatting
            if not conversation.strip():
                logger.warning("Empty conversation after formatting transcript data")
                return self._get_default_comprehensive_analysis_result()
            
            # Perform comprehensive analysis in a single API call for efficiency
            comprehensive_analysis = await self._perform_comprehensive_analysis(conversation, system_prompt)
            
            # Extract individual results from comprehensive analysis
            call_success = comprehensive_analysis.get("call_success", False)
            call_summary = comprehensive_analysis.get("call_summary", "")
            sentiment_score = comprehensive_analysis.get("sentiment_score", 0.0)
            detailed_analysis = comprehensive_analysis.get("detailed_analysis", {})
            
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
    
    
    async def _perform_comprehensive_analysis(self, conversation: str, system_prompt: str) -> Dict[str, Any]:
        """
        Perform comprehensive call analysis in a single API call.
        
        Args:
            conversation: Formatted conversation text
            system_prompt: System prompt for context
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            prompt = f"""
Analyze the following call conversation and provide a comprehensive analysis in JSON format.

System Context: {system_prompt}

Conversation:
{conversation}

Please provide a JSON response with the following structure:
{{
    "call_success": boolean,
    "call_summary": "Brief summary of the call",
    "sentiment_score": float (0.0 to 1.0, where 1.0 is most positive),
    "detailed_analysis": {{
        "key_topics": ["topic1", "topic2"],
        "customer_satisfaction": "high/medium/low",
        "resolution_status": "resolved/partially_resolved/not_resolved",
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
}}

Guidelines:
- call_success: true if the customer's main concern was addressed
- call_summary: 1-2 sentences summarizing the call outcome
- sentiment_score: 0.0 (negative) to 1.0 (positive)
- key_topics: Main topics discussed
- customer_satisfaction: Overall customer satisfaction level
- resolution_status: Whether the customer's issue was resolved
- improvement_suggestions: Specific suggestions for better call handling
"""

            response_text = await self._make_openai_request(prompt, max_tokens=500)
            
            # Parse JSON response - handle potential formatting issues
            import json
            import re
            
            try:
                # Clean the response text to extract JSON
                cleaned_text = response_text.strip()
                
                # Try to find JSON object in the response
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    analysis_result = json.loads(json_str)
                    return analysis_result
                else:
                    # Try parsing the entire response as JSON
                    analysis_result = json.loads(cleaned_text)
                    return analysis_result
                    
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse comprehensive analysis JSON: {e}")
                logger.warning(f"Raw response: {response_text[:200]}...")
                return self._get_default_comprehensive_analysis()
                
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._get_default_comprehensive_analysis()
    
    def _get_default_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get default comprehensive analysis result."""
        return {
            "call_success": False,
            "call_summary": "Analysis unavailable",
            "sentiment_score": 0.5,
            "detailed_analysis": {
                "key_topics": [],
                "customer_satisfaction": "unknown",
                "resolution_status": "unknown",
                "improvement_suggestions": []
            }
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
            if not self.client:
                # Fallback to direct API call if client isn't initialized
                import openai
                client = openai.AsyncClient(api_key=settings.openai_api_key)
            else:
                client = self.client
                
            response = await client.chat.completions.create(
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
    
    def _get_default_comprehensive_analysis_result(self) -> CallAnalysisResult:
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
                "key_topics": [],
                "customer_satisfaction": "unknown",
                "resolution_status": "unknown",
                "improvement_suggestions": ["Ensure proper transcript capture for future calls"]
            }
        )


# Global instance
call_analysis_service = CallAnalysisService()
