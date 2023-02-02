import os
from fastapi import APIRouter, status, BackgroundTasks
from ..engine import Config
from ..schemas.log import log_transaction
from ..dependencies import DATA_CACHE

router = APIRouter()


@router.post(
    '/config',
    response_model=Config,
    status_code=status.HTTP_201_CREATED,
)
async def configuration_engine(config: Config,
                               background_task: BackgroundTasks):
    DATA_CACHE.update(config.dict())
    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/setting/config',
        payload=config.dict()
    )
    return config


@router.get('/config', response_model=Config)
async def get_configuration(background_task: BackgroundTasks):
    background_task.add_task(
        log_transaction,
        method='/GET',
        endpoint='/setting/config',
    )
    return DATA_CACHE
