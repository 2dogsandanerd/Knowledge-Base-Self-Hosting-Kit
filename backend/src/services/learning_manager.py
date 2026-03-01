from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.database.models import LearningPair, User
from typing import Optional, List
from loguru import logger


class LearningManager:
    """
    Async Learning Manager for draft-sent email pairs.
    """

    def __init__(self, db: AsyncSession):
        """Initialize with async session"""
        self.db = db

    async def add_draft(
        self,
        user_id: int,
        thread_id: str,
        draft_message_id: str,
        draft_content: str,
        status: str = 'DRAFT_CREATED'
    ) -> int:
        """Async add draft to learning database"""
        try:
            learning_pair = LearningPair(
                user_id=user_id,
                thread_id=thread_id,
                draft_message_id=draft_message_id,
                draft_content=draft_content,
                status=status
            )

            self.db.add(learning_pair)
            await self.db.commit()
            await self.db.refresh(learning_pair)

            logger.info(f"Draft added to learning DB: ID={learning_pair.id}, thread={thread_id}")
            return learning_pair.id

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to add draft: {e}")
            return -1

    async def get_all_pairs(self, user_id: int) -> List[LearningPair]:
        """Async get all learning pairs for user"""
        stmt = select(LearningPair).where(
            LearningPair.user_id == user_id
        ).order_by(LearningPair.created_at.desc())

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_pair_by_draft_id(self, draft_message_id: str) -> Optional[LearningPair]:
        """Async get pair by draft message ID"""
        stmt = select(LearningPair).where(
            LearningPair.draft_message_id == draft_message_id
        )

        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_pair_by_thread_id(self, thread_id: str, user_id: int) -> Optional[LearningPair]:
        """Async get most recent pair for thread"""
        stmt = select(LearningPair).where(
            LearningPair.thread_id == thread_id,
            LearningPair.user_id == user_id,
            LearningPair.status != 'DELETED_NEGATIVE_EXAMPLE'
        ).order_by(LearningPair.created_at.desc())

        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_sent_email(
        self,
        draft_message_id: str,
        sent_message_id: str,
        sent_content: str
    ) -> bool:
        """Async update learning pair with sent email"""
        try:
            pair = await self.get_pair_by_draft_id(draft_message_id)

            if not pair:
                logger.warning(f"No learning pair found for draft {draft_message_id}")
                return False

            pair.sent_message_id = sent_message_id
            pair.sent_content = sent_content
            pair.status = 'PAIR_COMPLETED'

            await self.db.commit()

            logger.info(f"Learning pair updated: {pair.id}")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update sent email: {e}")
            return False

    async def is_thread_already_handled(self, thread_id: str, user_id: int) -> bool:
        """Async check if thread already has a draft"""
        pair = await self.get_pair_by_thread_id(thread_id, user_id)
        return pair is not None

    async def delete_pair(self, pair_id: int) -> bool:
        """Async delete learning pair"""
        try:
            stmt = select(LearningPair).where(LearningPair.id == pair_id)
            result = await self.db.execute(stmt)
            pair = result.scalar_one_or_none()

            if not pair:
                return False

            await self.db.delete(pair)
            await self.db.commit()

            logger.info(f"Learning pair {pair_id} deleted")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to delete pair: {e}")
            return False

    async def get_stats(self, user_id: int) -> dict:
        """Async get learning statistics"""
        try:
            # Total pairs
            stmt_total = select(func.count(LearningPair.id)).where(
                LearningPair.user_id == user_id
            )
            res_total = await self.db.execute(stmt_total)
            total_pairs = res_total.scalar() or 0

            # Completed pairs
            stmt_completed = select(func.count(LearningPair.id)).where(
                LearningPair.user_id == user_id,
                LearningPair.status == 'PAIR_COMPLETED'
            )
            res_completed = await self.db.execute(stmt_completed)
            completed_pairs = res_completed.scalar() or 0

            # Pending drafts
            stmt_pending = select(func.count(LearningPair.id)).where(
                LearningPair.user_id == user_id,
                LearningPair.status == 'DRAFT_CREATED'
            )
            res_pending = await self.db.execute(stmt_pending)
            pending_drafts = res_pending.scalar() or 0

            return {
                "total_pairs": total_pairs,
                "completed_pairs": completed_pairs,
                "pending_drafts": pending_drafts
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_pairs": 0,
                "completed_pairs": 0,
                "pending_drafts": 0
            }

    async def match_sent_emails(self, email_client, user_id: int) -> dict:
        """Async match sent emails with pending drafts"""
        # Get pending drafts
        stmt_pending = select(LearningPair).where(
            LearningPair.user_id == user_id,
            LearningPair.status == 'DRAFT_CREATED'
        )
        res_pending = await self.db.execute(stmt_pending)
        pending_drafts = res_pending.scalars().all()

        if not pending_drafts:
            return {"checked": 0, "completed": 0, "errors": 0}

        checked = 0
        completed = 0
        errors = 0

        try:
            # Get user email
            user_stmt = select(User).where(User.id == user_id)
            user_res = await self.db.execute(user_stmt)
            user = user_res.scalar_one_or_none()
            if not user:
                raise Exception("User not found")

            # Fetch sent emails
            sent_emails = await email_client.get_emails(folder_name="Sent", max_count=50)

            for draft in pending_drafts:
                checked += 1
                matching_sent = None
                for sent in sent_emails:
                    if sent.get('thread_id') == draft.thread_id:
                        matching_sent = sent
                        break
                
                if matching_sent:
                    success = await self.update_sent_email(
                        draft.draft_message_id,
                        matching_sent.get('id'),
                        matching_sent.get('body', '')
                    )
                    if success:
                        completed += 1
                    else:
                        errors += 1

        except Exception as e:
            logger.error(f"Error matching sent emails: {e}")
            errors += 1

        report = {"checked": checked, "completed": completed, "errors": errors}
        logger.info(f"Matching complete: {report}")
        return report