"""
Redis cache implementation for LLM Gateway
Provides caching for chat completions to reduce latency and cost
"""

import redis.asyncio as redis
import json
from typing import Optional, Any
from loguru import logger


class RedisCache:
    """Async Redis cache manager"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis cache

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
        """
        self.host = host
        self.port = port
        self.db = db
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis server"""
        try:
            self.client = await redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True
            )
            await self.client.ping()
            logger.info(f"✅ Redis connected: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis server"""
        if self.client:
            await self.client.close()
            logger.info("✅ Redis disconnected")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.client:
            return None

        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"⚠️ Cache get error for key '{key}': {e}")

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        if not self.client:
            return

        try:
            await self.client.setex(
                key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.warning(f"⚠️ Cache set error for key '{key}': {e}")

    async def delete(self, key: str):
        """
        Delete value from cache

        Args:
            key: Cache key
        """
        if not self.client:
            return

        try:
            await self.client.delete(key)
        except Exception as e:
            logger.warning(f"⚠️ Cache delete error for key '{key}': {e}")

    async def clear_pattern(self, pattern: str):
        """
        Clear all keys matching pattern

        Args:
            pattern: Redis key pattern (e.g., "chat:*")
        """
        if not self.client:
            return

        try:
            deleted_count = 0
            async for key in self.client.scan_iter(match=pattern):
                await self.client.delete(key)
                deleted_count += 1
            logger.info(f"✅ Cleared {deleted_count} keys matching pattern: {pattern}")
        except Exception as e:
            logger.warning(f"⚠️ Cache clear error for pattern '{pattern}': {e}")

    async def get_stats(self) -> dict:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        if not self.client:
            return {}

        try:
            info = await self.client.info('stats')
            return {
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'keys': await self.client.dbsize()
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cache stats: {e}")
            return {}
