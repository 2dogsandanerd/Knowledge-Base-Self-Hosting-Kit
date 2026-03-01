from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.database.models import Conversation
from typing import Optional, List
import uuid
import json
from loguru import logger


class ConversationManager:
    """Async Conversation Manager for conversation history"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def store_conversation(
        self,
        user_id: int,
        email_data: dict,
        generated_response: str,
        rag_context: str,
        model_used: str
    ) -> str:
        """Async store conversation"""
        try:
            conv_id = str(uuid.uuid4())

            conversation = Conversation(
                id=conv_id,
                user_id=user_id,
                email_data=json.dumps(email_data),
                generated_response=generated_response,
                rag_context_used=rag_context,
                model_used=model_used
            )

            self.db.add(conversation)
            await self.db.commit()

            logger.info(f"Conversation stored: {conv_id}")
            return conv_id

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to store conversation: {e}")
            return ""

    async def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Async get conversation by ID"""
        stmt = select(Conversation).where(Conversation.id == conv_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_conversations(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[Conversation]:
        """Async get all conversations for user"""
        stmt = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(
            Conversation.created_at.desc()
        ).limit(limit)

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def delete_conversation(self, conv_id: str) -> bool:
        """Async delete conversation"""
        try:
            stmt = select(Conversation).where(Conversation.id == conv_id)
            result = await self.db.execute(stmt)
            conv = result.scalar_one_or_none()

            if not conv:
                return False

            await self.db.delete(conv)
            await self.db.commit()

            logger.info(f"Conversation {conv_id} deleted")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to delete conversation: {e}")
            return False