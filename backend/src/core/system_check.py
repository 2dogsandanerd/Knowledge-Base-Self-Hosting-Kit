import shutil
import os
import aiohttp
from typing import Dict, Any, List
from loguru import logger
from pathlib import Path

class SystemHealthCheck:
    """
    Comprehensive system health check for the Onboarding Wizard.
    Checks disk space, permissions, and AI service availability.
    """

    @staticmethod
    def check_disk_space(path: str = ".", min_gb: int = 2) -> Dict[str, Any]:
        """Check if there is enough free disk space."""
        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (2**30)
            
            status = "ok" if free_gb >= min_gb else "warning"
            if free_gb < 0.5: # Critical if less than 500MB
                status = "critical"
                
            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "required_gb": min_gb,
                "message": f"Free space: {round(free_gb, 2)} GB"
            }
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def check_permissions(paths: List[str]) -> Dict[str, Any]:
        """Check read/write permissions for critical directories."""
        results = {}
        all_ok = True
        
        for path_str in paths:
            path = Path(path_str)
            # Create if not exists to test creation
            try:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                
                # Test write
                test_file = path / ".perm_test"
                test_file.touch()
                test_file.unlink()
                
                results[path_str] = "ok"
            except Exception as e:
                logger.error(f"Permission check failed for {path_str}: {e}")
                results[path_str] = "error"
                all_ok = False
                
        return {
            "status": "ok" if all_ok else "error",
            "details": results
        }

    @staticmethod
    async def check_ollama(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
        """Check if Ollama is running and list models."""
        try:
            async with aiohttp.ClientSession() as session:
                # Check version/status
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        return {
                            "status": "ok",
                            "available": True,
                            "models": models,
                            "count": len(models)
                        }
                    else:
                        return {
                            "status": "error",
                            "available": False,
                            "message": f"Ollama returned status {response.status}"
                        }
        except aiohttp.ClientConnectorError:
            return {
                "status": "critical",
                "available": False,
                "message": "Connection refused. Is Ollama running?"
            }
        except Exception as e:
            return {
                "status": "error",
                "available": False,
                "message": str(e)
            }
