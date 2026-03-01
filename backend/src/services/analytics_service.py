from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import csv
import io

from src.database.models import LearningPair, User


class AnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def calculate_engagement_score(self) -> int:
        """Calculate customer engagement score (0-100)."""
        try:
            # Get total learning pairs
            total_pairs_stmt = select(func.count(LearningPair.id))
            total_pairs_res = await self.db.execute(total_pairs_stmt)
            total_pairs = total_pairs_res.scalar() or 0
            
            if total_pairs == 0:
                return 0
            
            # Get completed pairs (successful learning)
            completed_pairs_stmt = select(func.count(LearningPair.id)).where(
                LearningPair.status == "PAIR_COMPLETED"
            )
            completed_pairs_res = await self.db.execute(completed_pairs_stmt)
            completed_pairs = completed_pairs_res.scalar() or 0
            
            # Get recent activity (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_pairs_stmt = select(func.count(LearningPair.id)).where(
                LearningPair.created_at >= week_ago
            )
            recent_pairs_res = await self.db.execute(recent_pairs_stmt)
            recent_pairs = recent_pairs_res.scalar() or 0
            
            # Calculate base score from completion rate
            completion_rate = (completed_pairs / total_pairs) * 100 if total_pairs > 0 else 0
            
            # Bonus for recent activity
            activity_bonus = min(recent_pairs * 2, 20)  # Max 20 points for activity
            
            # Final score (0-100)
            score = min(int(completion_rate + activity_bonus), 100)
            
            return max(score, 0)
        except Exception:
            return 0

    def get_effort_level(self, score: int) -> str:
        """Get effort level based on engagement score."""
        if score >= 80:
            return "Low Effort"
        elif score >= 60:
            return "Medium Effort"
        elif score >= 40:
            return "High Effort"
        else:
            return "Critical"

    async def get_total_conversations(self) -> int:
        """Get total number of conversations."""
        stmt = select(func.count(LearningPair.id))
        res = await self.db.execute(stmt)
        return res.scalar() or 0

    async def get_avg_response_time(self) -> float:
        """Get average response time in hours."""
        try:
            # Calculate average time between email and draft creation
            stmt = select(LearningPair).where(LearningPair.status == "PAIR_COMPLETED")
            res = await self.db.execute(stmt)
            pairs = res.scalars().all()
            
            if not pairs:
                return 0.0
            
            total_hours = 0
            valid_pairs = 0
            
            for pair in pairs:
                if pair.created_at and pair.updated_at:
                    time_diff = pair.updated_at - pair.created_at
                    hours = time_diff.total_seconds() / 3600
                    total_hours += hours
                    valid_pairs += 1
            
            return round(total_hours / valid_pairs, 1) if valid_pairs > 0 else 0.0
        except Exception:
            return 0.0

    async def get_reply_rate(self) -> float:
        """Get reply rate percentage."""
        try:
            total_pairs_stmt = select(func.count(LearningPair.id))
            total_pairs_res = await self.db.execute(total_pairs_stmt)
            total_pairs = total_pairs_res.scalar() or 0

            if total_pairs == 0:
                return 0.0
            
            completed_pairs_stmt = select(func.count(LearningPair.id)).where(
                LearningPair.status == "PAIR_COMPLETED"
            )
            completed_pairs_res = await self.db.execute(completed_pairs_stmt)
            completed_pairs = completed_pairs_res.scalar() or 0
            
            return round((completed_pairs / total_pairs) * 100, 1)
        except Exception:
            return 0.0

    async def get_avg_conversation_length(self) -> float:
        """Get average conversation length (messages per thread)."""
        try:
            stmt = select(LearningPair).where(LearningPair.status == "PAIR_COMPLETED")
            res = await self.db.execute(stmt)
            pairs = res.scalars().all()
            
            if not pairs:
                return 0.0
            
            total_length = 0
            valid_pairs = 0
            
            for pair in pairs:
                if pair.draft_content:
                    estimated_messages = max(1, len(pair.draft_content.split('\n')) // 3)
                    total_length += estimated_messages
                    valid_pairs += 1
            
            return round(total_length / valid_pairs, 1) if valid_pairs > 0 else 0.0
        except Exception:
            return 0.0

    async def get_daily_learning_counts(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily learning pair counts for the last N days."""
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            stmt = select(
                func.date(LearningPair.created_at).label('date'),
                func.count(LearningPair.id).label('count')
            ).where(
                and_(
                    func.date(LearningPair.created_at) >= start_date,
                    func.date(LearningPair.created_at) <= end_date
                )
            ).group_by(
                func.date(LearningPair.created_at)
            ).order_by('date')

            res = await self.db.execute(stmt)
            daily_counts = res.all()
            
            result = []
            current_date = start_date
            
            while current_date <= end_date:
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

    def calculate_trend(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate linear regression trend."""
        try:
            if len(data) < 2:
                return {"slope": 0, "r_squared": 0}
            
            n = len(data)
            x_values = list(range(n))
            y_values = [d["count"] for d in data]
            
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return {"slope": 0, "r_squared": 0}
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            y_pred = [slope * x + intercept for x in x_values]
            ss_res = sum((y - pred) ** 2 for y, pred in zip(y_values, y_pred))
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                "slope": round(slope, 3),
                "r_squared": round(r_squared, 3)
            }
        except Exception:
            return {"slope": 0, "r_squared": 0}

    async def get_conversation_by_status(self) -> Dict[str, int]:
        """Get conversation counts by status."""
        try:
            stmt = select(
                LearningPair.status,
                func.count(LearningPair.id).label('count')
            ).group_by(LearningPair.status)

            res = await self.db.execute(stmt)
            status_counts = res.all()
            
            result = {}
            for status, count in status_counts:
                result[status] = count
            
            return result
        except Exception:
            return {}

    async def generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate AI-powered learning recommendations."""
        try:
            recommendations = []
            
            engagement_score = await self.calculate_engagement_score()
            total_conversations = await self.get_total_conversations()
            reply_rate = await self.get_reply_rate()
            avg_response_time = await self.get_avg_response_time()
            
            if engagement_score < 40:
                recommendations.append({
                    "type": "focus",
                    "message": "🎯 Focus on improving response quality. Your engagement score is low - consider reviewing draft templates."
                })
            elif engagement_score >= 80:
                recommendations.append({
                    "type": "success",
                    "message": "✅ Good job on maintaining high engagement! Keep up the excellent work."
                })
            
            if reply_rate < 50:
                recommendations.append({
                    "type": "warning",
                    "message": "⚠️ Warning: Low reply rate detected. Consider improving email personalization."
                })
            
            if avg_response_time > 24:
                recommendations.append({
                    "type": "focus",
                    "message": "🎯 Focus on reducing response time. Consider automating common responses."
                })
            
            if total_conversations < 10:
                recommendations.append({
                    "type": "info",
                    "message": "ℹ️ Build more conversation history to improve AI recommendations."
                })
            
            return recommendations
        except Exception:
            return []

    async def export_to_csv(self) -> str:
        """Export learning data to CSV format."""
        try:
            stmt = select(LearningPair)
            res = await self.db.execute(stmt)
            pairs = res.scalars().all()
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow([
                'ID', 'Thread ID', 'Status', 'Created At', 'Updated At', 
                'Draft Content', 'User ID'
            ])
            
            for pair in pairs:
                writer.writerow([
                    pair.id,
                    pair.thread_id,
                    pair.status,
                    pair.created_at.isoformat() if pair.created_at else '',
                    pair.updated_at.isoformat() if pair.updated_at else '',
                    pair.draft_content or '',
                    pair.user_id
                ])
            
            return output.getvalue()
        except Exception:
            return ""

    async def export_to_json(self) -> str:
        """Export learning data to JSON format."""
        try:
            stmt = select(LearningPair)
            res = await self.db.execute(stmt)
            pairs = res.scalars().all()
            
            data = {
                "export_date": datetime.utcnow().isoformat(),
                "total_pairs": len(pairs),
                "pairs": []
            }
            
            for pair in pairs:
                data["pairs"].append({
                    "id": pair.id,
                    "thread_id": pair.thread_id,
                    "status": pair.status,
                    "created_at": pair.created_at.isoformat() if pair.created_at else None,
                    "updated_at": pair.updated_at.isoformat() if pair.updated_at else None,
                    "draft_content": pair.draft_content,
                    "user_id": pair.user_id
                })
            
            return json.dumps(data, indent=2)
        except Exception:
            return "{}"

    async def reset_all_data(self) -> int:
        """Reset all learning data (DANGEROUS)."""
        try:
            stmt = select(func.count(LearningPair.id))
            res = await self.db.execute(stmt)
            count = res.scalar() or 0
            
            delete_stmt = LearningPair.__table__.delete()
            await self.db.execute(delete_stmt)
            await self.db.commit()
            
            return count
        except Exception:
            await self.db.rollback()
            return 0