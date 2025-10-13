"""
FastAPI application factory.
Creates and configures the main FastAPI application instance.
"""

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.core.logging import setup_logging
from app.api.routes import call_routes, call_logs_routes, websocket_routes, assistant_routes, inbound_routes, organization_routes, auth_routes, knowledge_routes, dashboard_routes, rag_routes, voice_routes, webrtc_routes, campaign_routes, tool_routes
# Removed old call_handler - using unified_pipeline_manager now
from app.db.database import init_db

# Set up logging
setup_logging()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: The configured application instance
    """
    
    app = FastAPI(
        title="Voice Assistant API",
        description="API for creating AI-powered voice calls",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """
        Initialize services and database when the application starts.
        
        This ensures all services are pre-loaded and ready before
        the first call, reducing latency when users answer the phone.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("🚀 Starting service and database initialization...")
            
            # Initialize database
            await init_db()
            
            # Initialize async call service
            from app.services.async_call_service import async_call_service
            await async_call_service.start()
            logger.info("✅ Async call service started")
            
            # Initialize all services in the background
            # This runs asynchronously so the server starts quickly
            # Services are initialized by unified_pipeline_manager when needed
            
            logger.info("✅ Service and database initialization completed")
        except Exception as e:
            # Log error but don't prevent server startup
            logger.error(f"❌ Failed to initialize services during startup: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """
        Clean up resources when the application shuts down.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Stop async call service
            from app.services.async_call_service import async_call_service
            await async_call_service.stop()
            logger.info("✅ Async call service stopped")
            
            # Close any active calls
            # Cleanup is handled by unified_pipeline_manager
            
            logger.info("All resources cleaned up")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # Include routers
    app.include_router(auth_routes.router)
    app.include_router(organization_routes.router)
    app.include_router(call_routes.router)
    app.include_router(call_logs_routes.router)
    app.include_router(websocket_routes.router)
    app.include_router(webrtc_routes.router)
    app.include_router(assistant_routes.router)
    app.include_router(inbound_routes.router)
    app.include_router(knowledge_routes.router)
    app.include_router(rag_routes.router)
    app.include_router(dashboard_routes.router)
    app.include_router(voice_routes.router)
    app.include_router(campaign_routes.router)
    app.include_router(tool_routes.router)
    # Removed duplicate call-status route; using /api/v1/call-status from call_routes
    
    return app


# Create the application instance
app = create_app()