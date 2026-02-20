from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from loguru import logger

# Base is already defined in models.py, but usually we define it here or import it.
# However, models.py defines Base = declarative_base().
# To avoid circular imports, we should probably import Base from models if we needed it for init_db,
# but for session creation we don't strictly need it.
# Let's just setup the engine and session factory.

# Default to SQLite for local development if not specified
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./sql_app.db")

logger.info(f"Initializing database connection to: {DATABASE_URL}")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # check_same_thread is needed for SQLite
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db():
    """Dependency for getting async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
