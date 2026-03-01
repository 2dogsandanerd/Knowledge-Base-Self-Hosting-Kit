
import time
import httpx
import asyncio
from typing import Dict, Any, List
from loguru import logger

# Assuming clients or connection logic will be available from other services
# This is a simplified adaptation of Project B's service.

class ConnectionService:
    """Provides methods to test connections to external services."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.logger = logger.bind(component="ConnectionService")

    async def test_ollama_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Tests connection to Ollama."""
        start_time = time.time()
        host = config.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{host}/api/tags", timeout=self.timeout)
                response.raise_for_status()
                duration = time.time() - start_time
                return {
                    "success": True,
                    "message": f"Ollama connection to {host} successful.",
                    "duration": duration
                }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Ollama connection test failed: {e}")
            return {"success": False, "message": str(e), "duration": duration}

    async def test_chroma_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Tests connection to ChromaDB via ChromaManager."""
        start_time = time.time()

        try:
            # Use the centralized ChromaManager for connection testing
            from src.core.chroma_manager import get_chroma_manager
            chroma_manager = get_chroma_manager()

            # Use async method for proper connection testing
            client = await chroma_manager.get_client_async()

            if client:
                # Test with heartbeat (wrap in asyncio.to_thread for sync operation)
                heartbeat = await asyncio.to_thread(client.heartbeat)
                duration = time.time() - start_time
                self.logger.info(f"ChromaDB heartbeat: {heartbeat}")
                return {
                    "success": True,
                    "message": f"ChromaDB connection via ChromaManager successful.",
                    "duration": duration
                }
            else:
                duration = time.time() - start_time
                self.logger.error("ChromaDB client not available via ChromaManager")
                return {
                    "success": False,
                    "message": "ChromaDB client not available via ChromaManager",
                    "duration": duration
                }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"ChromaDB connection test failed: {e}")
            return {"success": False, "message": str(e), "duration": duration}

    async def test_knowledge_base_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Tests connection to Knowledge Base API."""
        start_time = time.time()
        
        api_url = config.get("KNOWLEDGE_BASE_API_URL", "http://localhost:8080")
        timeout_ms = config.get("KNOWLEDGE_BASE_TIMEOUT", "120000")
        timeout_sec = int(timeout_ms) / 1000
        
        try:
            # Ensure the URL ends with /health for the health check
            health_url = api_url.rstrip('/') + '/health'
            
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    duration = time.time() - start_time
                    data = response.json()
                    message = f"Knowledge Base API connection successful. Status: {data.get('status', 'unknown')}"
                    return {
                        "success": True,
                        "message": message,
                        "duration": duration
                    }
                else:
                    duration = time.time() - start_time
                    return {
                        "success": False,
                        "message": f"Knowledge Base API responded with status {response.status_code}",
                        "duration": duration
                    }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Knowledge Base API connection test failed: {e}")
            return {"success": False, "message": f"Knowledge Base API connection failed: {str(e)}", "duration": duration}

    async def test_all_connections(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Tests all relevant connections concurrently."""
        self.logger.info("Testing all connections...")
        tasks = {
            "ollama": self.test_ollama_connection(config),
            "chroma": self.test_chroma_connection(config),
            "knowledge_base_api": self.test_knowledge_base_api(config),
            # Add other tests like email here in the future
        }

        results = await asyncio.gather(*tasks.values())

        final_results = []
        for i, (component, _) in enumerate(tasks.items()):
            result = results[i]
            result["component"] = component
            final_results.append(result)

        return final_results

connection_service = ConnectionService()
