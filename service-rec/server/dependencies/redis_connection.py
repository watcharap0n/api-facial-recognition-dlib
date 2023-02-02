"""
redis for the connection then started server.

:main.py dependencies
"""

import os
from aioredis import Redis


async def init_redis():
    redis_conn = await Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0
    )
    return redis_conn
