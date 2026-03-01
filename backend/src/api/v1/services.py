from src.core.exceptions import ServiceUnavailableError, ValidationError
from fastapi import APIRouter, HTTPException, Body, WebSocket, WebSocketDisconnect
from typing import Dict, Any
from datetime import datetime
import asyncio
from loguru import logger

from src.services.service_manager import service_manager
from src.services.config_service import config_service
from src.services.health_monitor import get_health_monitor
from src.core.config import get_websocket_config

router = APIRouter()



@router.get("/status")
async def get_services_status():
    """Get status of all services."""
    try:
        config = config_service.load_configuration()
        status = await service_manager.get_status(config)
        return status
    except Exception as e:
        raise ServiceUnavailableError("service", str(e))


@router.get("/ollama/models")
async def get_ollama_models():
    """Get available Ollama models."""
    try:
        config = config_service.load_configuration()
        ollama_host = config.get("OLLAMA_HOST", "http://localhost:11434")

        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return {"models": data.get("models", [])}
            else:
                logger.error(f"Ollama returned status {response.status_code}")
                raise ServiceUnavailableError("service", "Ollama not reachable")
    except httpx.TimeoutException:
        logger.error("Ollama request timed out")
        raise ServiceUnavailableError("service", "Ollama request timed out")
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {e}")
        raise ServiceUnavailableError("service", str(e))


@router.websocket("/ws/status")
async def websocket_status_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time service status updates with backpressure handling."""
    ws_config = get_websocket_config()
    monitor = get_health_monitor()

    await websocket.accept()
    logger.info(f"WebSocket client connected: {websocket.client}")

    if not monitor.can_subscribe():
        await websocket.close(code=1008, reason="Maximum subscribers reached")
        return

    # Bounded queue for backpressure management
    queue = asyncio.Queue(maxsize=ws_config.max_queue_size)

    # Backpressure-safe callback wrapper
    def on_status_event(event: Dict[str, Any]):
        """Handle status events with drop-oldest strategy on backpressure."""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event and try again
            try:
                _ = queue.get_nowait()
                queue.put_nowait(event)
                logger.warning(f"WS backpressure: dropped oldest event for {websocket.client}")
            except Exception as e:
                logger.warning(f"WS event dropped due to backpressure: {e}")

    # Subscribe to monitor with backpressure-safe callback
    monitor.subscribe(on_status_event)
    last_activity = datetime.now()

    try:
        # Send initial status
        config = config_service.load_configuration()
        current_status = await service_manager.get_status(config)
        await websocket.send_json({
            "type": "initial_status",
            "data": current_status,
            "timestamp": datetime.now().isoformat()
        })

        # Listen for updates with graceful disconnection handling
        while True:
            # Calculate timeout for idle detection (0 = disabled)
            timeout = ws_config.idle_timeout if ws_config.idle_timeout > 0 else None

            try:
                # Use asyncio.wait with timeout support
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(queue.get()),
                        asyncio.create_task(websocket.receive_text())
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout
                )

                # Check for idle timeout
                if not done:
                    # Timeout occurred - no events or client messages
                    elapsed = (datetime.now() - last_activity).total_seconds()
                    if timeout and elapsed >= timeout:
                        logger.info(f"WS idle timeout ({timeout}s) for {websocket.client}")
                        await websocket.close(code=1000, reason="Idle timeout")
                        break
                    # Cancel pending and continue
                    for task in pending:
                        task.cancel()
                    continue

                # Update last activity
                last_activity = datetime.now()

                # Get the result that completed first
                for task in done:
                    result = task.result()

                    # If it's from queue.get(), it's a status update event (dict)
                    if isinstance(result, dict):
                        await websocket.send_json({
                            "type": "status_change",
                            "data": result,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # Message from client (keepalive or control message)
                        try:
                            msg = str(result).strip().lower()
                        except Exception:
                            msg = ""
                        if msg == "ping":
                            await websocket.send_text("pong")
                            logger.debug(f"WS pong sent to {websocket.client}")
                        # Do NOT break here — keep the connection open

                # Cancel pending tasks to avoid leaks
                for task in pending:
                    task.cancel()

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {websocket.client}")
                break
            except Exception as e:
                # Log unexpected errors and break out to ensure cleanup
                logger.error(f"WebSocket loop error: {e}", exc_info=True)
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            # Always unsubscribe from monitor to prevent memory leaks
            monitor.unsubscribe(on_status_event)
            logger.info(f"WebSocket client unsubscribed (subscribers: {len(monitor._subscribers)})")
        except Exception as e:
            logger.error(f"Error unsubscribing WebSocket client: {e}")