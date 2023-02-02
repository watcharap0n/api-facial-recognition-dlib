import logging
import pytz
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
handler = logging.FileHandler('log.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


class StatusTraining(BaseModel):
    training_method: Optional[str] = None
    training_status_success: bool
    training_progress: Optional[str] = '0.0'

    class Config:
        schema_extra = {
            'example': {
                'training_method': 'str',
                'training_status_success': False,
                'training_progress': '0.0 %'
            }
        }


async def log_transaction(
        method: str,
        endpoint: str,
        payload: Optional[dict] = None,
):
    tz = pytz.timezone('Asia/Bangkok')
    data = {
        'method': method,
        'endpoint': endpoint,
        'payload': payload,
        'datetime': datetime.now(tz)
    }
    LOGGER.info(data)
