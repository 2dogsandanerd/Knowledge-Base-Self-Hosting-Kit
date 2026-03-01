from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from src.database.models import LearningPair, Conversation
from datetime import datetime, timedelta
from typing import Dict, Any
from loguru import logger


class StatisticsService:
    """Async Statistics Service"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard_stats(self, user_id: int) -> Dict[str, Any]:
        """Async get dashboard statistics"""
        try:
            # Total drafts
            stmt_drafts = select(func.count(LearningPair.id)).where(
                LearningPair.user_id == user_id
            )
            result_drafts = await self.db.execute(stmt_drafts)
            total_drafts = result_drafts.scalar() or 0

            # Sent emails
            stmt_sent = select(func.count(LearningPair.id)).where(
                and_(
                    LearningPair.user_id == user_id,
                    LearningPair.status == 'PAIR_COMPLETED'
                )
            )
            result_sent = await self.db.execute(stmt_sent)
            sent_count = result_sent.scalar() or 0

            # Total conversations
            stmt_convs = select(func.count(Conversation.id)).where(
                Conversation.user_id == user_id
            )
            result_convs = await self.db.execute(stmt_convs)
            total_conversations = result_convs.scalar() or 0

            # Drafts today
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            stmt_today = select(func.count(LearningPair.id)).where(
                and_(
                    LearningPair.user_id == user_id,
                    LearningPair.created_at >= today_start
                )
            )
            result_today = await self.db.execute(stmt_today)
            drafts_today = result_today.scalar() or 0

            return {
                'total_drafts': total_drafts,
                'sent_count': sent_count,
                'total_conversations': total_conversations,
                'drafts_today': drafts_today,
                'conversion_rate': (sent_count / total_drafts * 100) if total_drafts > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to get dashboard stats: {e}")
            return {
                'total_drafts': 0,
                'sent_count': 0,
                'total_conversations': 0,
                'drafts_today': 0,
                'conversion_rate': 0
            }

    async def get_weekly_stats(self, user_id: int) -> Dict[str, Any]:
        """Async get weekly statistics"""
        try:
            week_ago = datetime.now() - timedelta(days=7)

            # Drafts last 7 days
            stmt = select(
                func.date(LearningPair.created_at).label('date'),
                func.count(LearningPair.id).label('count')
            ).where(
                and_(
                    LearningPair.user_id == user_id,
                    LearningPair.created_at >= week_ago
                )
            ).group_by(func.date(LearningPair.created_at))

            result = await self.db.execute(stmt)
            rows = result.all()

            daily_stats = {str(row.date): row.count for row in rows}

            return {
                'daily_drafts': daily_stats,
                'total_week': sum(daily_stats.values())
            }

        except Exception as e:
            logger.error(f"Failed to get weekly stats: {e}")
            return {'daily_drafts': {}, 'total_week': 0}

    async def get_daily_email_counts(self, user_id: int, days: int) -> Dict[str, Any]:
        """Async get daily email counts for a specified number of days."""
        try:
            # Calculate the start date for the query
            start_date = datetime.now() - timedelta(days=days)

            # Query to get daily counts of LearningPair for the last 'days'
            stmt = select(
                func.date(LearningPair.created_at).label('date'),
                func.count(LearningPair.id).label('count')
            ).where(
                and_(
                    LearningPair.user_id == user_id,
                    LearningPair.created_at >= start_date
                )
            ).group_by(func.date(LearningPair.created_at))

            result = await self.db.execute(stmt)
            rows = result.all()

            # Format the results into a dictionary { "YYYY-MM-DD": count }
            daily_counts = {str(row.date): row.count for row in rows}

            # Ensure all 'days' in the range are present, even if count is 0
            all_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
            
            # Fill in missing dates with a count of 0
            for d in all_dates:
                if d not in daily_counts:
                    daily_counts[d] = 0

            # Sort by date
            sorted_daily_counts = dict(sorted(daily_counts.items()))

            return sorted_daily_counts

        except Exception as e:
            logger.error(f"Failed to get daily email counts: {e}")
            return {}