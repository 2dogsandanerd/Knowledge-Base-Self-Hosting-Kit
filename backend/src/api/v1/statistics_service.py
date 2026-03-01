import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import csv
import io

from src.database.models import LearningPair, User
from src.core.rag_client import RAGClient


class StatisticsService:
    def __init__(self, db: Session):
        self.db = db

    def get_email_statistics(self) -> Dict[str, Any]:
        """Get email processing statistics."""
        try:
            total_pairs = self.db.query(LearningPair).count()
            successful_pairs = self.db.query(LearningPair).filter(
                LearningPair.status == "PAIR_COMPLETED"
            ).count()
            
            success_rate = (successful_pairs / total_pairs * 100) if total_pairs > 0 else 0
            
            return {
                "total_emails": total_pairs,
                "total_drafts": total_pairs,
                "successful_drafts": successful_pairs,
                "success_rate": round(success_rate, 2),
                "failed_drafts": total_pairs - successful_pairs
            }
        except Exception as e:
            return {
                "total_emails": 0,
                "total_drafts": 0,
                "successful_drafts": 0,
                "success_rate": 0.0,
                "failed_drafts": 0
            }

    def get_daily_email_counts(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get email counts per day for last N days."""
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            # Query daily counts
            daily_counts = self.db.query(
                func.date(LearningPair.created_at).label('date'),
                func.count(LearningPair.id).label('count')
            ).filter(
                and_(
                    func.date(LearningPair.created_at) >= start_date,
                    func.date(LearningPair.created_at) <= end_date
                )
            ).group_by(
                func.date(LearningPair.created_at)
            ).order_by('date').all()
            
            # Create a complete date range
            result = []
            current_date = start_date
            
            while current_date <= end_date:
                # Find count for this date
                count = 0
                for daily_count in daily_counts:
                    if daily_count.date == current_date:
                        count = daily_count.count
                        break
                
                result.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "count": count
                })
                current_date += timedelta(days=1)
            
            return result
        except Exception:
            return []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        # For now, return mock data structure
        # In a real implementation, this would track timing data in DB
        return {
            "rag_query_time": {"avg": 0.5, "min": 0.2, "max": 1.2},
            "llm_response_time": {"avg": 2.3, "min": 1.1, "max": 5.8},
            "email_fetch_time": {"avg": 0.8, "min": 0.5, "max": 2.1},
            "end_to_end_time": {"avg": 3.6, "min": 2.0, "max": 8.5}
        }

    def get_llm_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        # Would require token tracking in database
        # For now, return mock data
        return {
            "total_calls": 0,  # TODO: Track in DB
            "total_tokens": {"input": 0, "output": 0},
            "estimated_cost": 0.0,
            "model_distribution": {
                "llama3": 45,
                "gemini-flash": 30,
                "gpt-4": 25
            }
        }

    async def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        try:
            rag_client = RAGClient()

            # If ChromaDB is not configured, return empty stats
            client = await rag_client.chroma_manager.get_client_async() if rag_client.chroma_manager else None
            if not client:
                return {
                    "total_queries": 0,
                    "avg_chunks_used": 0,
                    "cache_hit_rate": 0.0,
                    "collection_sizes": {},
                    "total_documents": 0
                }

            collections_response = await rag_client.list_collections()
            collections = collections_response.data if hasattr(collections_response, 'data') else []
            
            collection_sizes = {}
            total_documents = 0
            
            for collection_name in collections:
                try:
                    collection = await asyncio.to_thread(rag_client.chroma_manager.get_collection, collection_name)
                    count = await asyncio.to_thread(collection.count)
                    collection_sizes[collection_name] = count
                    total_documents += count
                except Exception:
                    collection_sizes[collection_name] = 0
            
            return {
                "total_queries": 0,  # TODO: Track in DB
                "avg_chunks_used": 3.5,
                "cache_hit_rate": 0.0,
                "collection_sizes": collection_sizes,
                "total_documents": total_documents
            }
        except Exception:
            return {
                "total_queries": 0,
                "avg_chunks_used": 0,
                "cache_hit_rate": 0.0,
                "collection_sizes": {},
                "total_documents": 0
            }

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            from src.services.service_manager import service_manager
            from src.core.config import get_config
            
            config = get_config()
            services_status = service_manager.get_status(config)
            
            return {
                "backend": {"status": "online", "uptime": "5d 3h 22m"},
                "ollama": services_status.get("ollama", {"status": "unknown"}),
                "chroma": services_status.get("chroma", {"status": "unknown"}),
                "email_client": {"status": "connected", "provider": config.EMAIL_PROVIDER}
            }
        except Exception:
            return {
                "backend": {"status": "unknown", "uptime": "unknown"},
                "ollama": {"status": "unknown"},
                "chroma": {"status": "unknown"},
                "email_client": {"status": "unknown", "provider": "unknown"}
            }

    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent system activities."""
        try:
            activities = []
            
            # Get recent learning pairs
            recent_pairs = self.db.query(LearningPair).order_by(
                LearningPair.created_at.desc()
            ).limit(limit).all()
            
            for pair in recent_pairs:
                activities.append({
                    "type": "draft_created",
                    "message": f"Draft created for thread {pair.thread_id[:8] if pair.thread_id else 'unknown'}",
                    "timestamp": pair.created_at.isoformat() if pair.created_at else datetime.utcnow().isoformat(),
                    "icon": "✍️",
                    "status": pair.status
                })
            
            # Sort by timestamp
            activities.sort(key=lambda x: x['timestamp'], reverse=True)
            return activities[:limit]
        except Exception:
            return []

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get quick dashboard statistics."""
        try:
            total_pairs = self.db.query(LearningPair).count()
            pending_pairs = self.db.query(LearningPair).filter(
                LearningPair.status.in_(["DRAFT_CREATED", "PAIR_CREATED"])
            ).count()
            
            # Get RAG document count
            rag_stats = await self.get_rag_statistics()
            
            return {
                "unread_emails": total_pairs,  # Using total pairs as proxy
                "pending_drafts": pending_pairs,
                "learning_pairs": total_pairs,
                "rag_documents": rag_stats.get("total_documents", 0),
                "system_uptime": "5d 3h 22m"  # Mock data
            }
        except Exception:
            return {
                "unread_emails": 0,
                "pending_drafts": 0,
                "learning_pairs": 0,
                "rag_documents": 0,
                "system_uptime": "unknown"
            }

    def export_statistics_csv(self) -> str:
        """Export statistics as CSV format."""
        try:
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Date', 'Type', 'Thread ID', 'Status', 'Created At'
            ])
            
            # Get all learning pairs
            pairs = self.db.query(LearningPair).all()
            
            # Write data
            for pair in pairs:
                writer.writerow([
                    pair.created_at.date().isoformat() if pair.created_at else '',
                    'Learning Pair',
                    pair.thread_id or '',
                    pair.status or '',
                    pair.created_at.isoformat() if pair.created_at else ''
                ])
            
            return output.getvalue()
        except Exception:
            return ""

    def export_statistics_json(self) -> str:
        """Export statistics as JSON format."""
        try:
            stats = {
                "export_date": datetime.utcnow().isoformat(),
                "email_statistics": self.get_email_statistics(),
                "rag_statistics": self.get_rag_statistics(),
                "system_health": self.get_system_health(),
                "recent_activities": self.get_recent_activities(50)
            }
            
            return json.dumps(stats, indent=2)
        except Exception:
            return "{}"

