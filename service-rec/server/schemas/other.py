from pydantic import BaseModel
from typing import Union


class SuccessRequest(BaseModel):
    status: bool
    detail: Union[dict, None] = None

    class Config:
        schema_extra = {
            'example': {
                'status': True,
                'detail': {
                    'message': 'your response success'
                }
            }
        }
