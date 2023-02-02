import os
import json
from fastapi import APIRouter, status, BackgroundTasks, Depends, HTTPException
from ..engine import Config, ReportConfig
from ..dependencies.redis_connection import init_redis
from ..schemas.report import ReportCropped
from ..schemas.log import log_transaction, StatusTraining

router = APIRouter()

DATA_CACHE = {}


@router.get(
    '/training',
    response_model=StatusTraining,
    status_code=status.HTTP_200_OK,
)
async def report_training_status(background_task: BackgroundTasks,
                                 redis_conn=Depends(init_redis)):
    background_task.add_task(
        log_transaction,
        method='/GET',
        endpoint='/report/training',
    )
    redis_model = await redis_conn.get('training_model')
    redis_model = json.loads(redis_model)
    if not redis_model:
        raise HTTPException(status_code=status.HTTP_200_OK,
                            detail='Not in progress train model.')
    return redis_model


@router.post('/cropped', response_model=ReportCropped)
async def report_cropped_images(config: ReportConfig):
    root_dir = os.path.abspath('.')
    try:
        datasets = os.path.join(root_dir, config.datasets_folder)
        details_missing = []
        for data in os.listdir(datasets):
            if not data.startswith('.'):
                path_id = os.path.join(datasets, data)
                path_label = os.path.join(path_id, config.labels_folder)
                path_ids = [i for i in os.listdir(path_id) if
                            i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg') or i.endswith('.JPG')]
                path_label_list = os.listdir(path_label)
                if len(path_ids) != len(path_label_list):
                    percent_success = (len(path_label_list) / len(path_ids)) * 100
                    error_crop = set(path_ids) - set(path_label_list)
                    response = {
                        'img_original': len(path_ids),
                        'cropped_success': len(path_label_list),
                        'id_folder': data,
                        'percentage': f'{percent_success} %',
                        'error_images': list(error_crop)
                    }
                    details_missing.append(response)
    except FileNotFoundError:
        raise HTTPException(detail='Folder not found error', status_code=status.HTTP_400_BAD_REQUEST)
    model_store = ReportCropped()
    if not details_missing:
        model_store.status = True
        return model_store
    model_store.status = False
    model_store.detail = details_missing
    return model_store


@router.get('/training/delete')
async def report_training_delete_status(redis_conn=Depends(init_redis)):
    await redis_conn.delete('training_model')
    return {'status': True, 'message': 'Success delete memory cache.'}
