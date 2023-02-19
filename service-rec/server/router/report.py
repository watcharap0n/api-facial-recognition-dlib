import os
import json
from fastapi import APIRouter, status, BackgroundTasks, Depends, HTTPException
from ..engine import ReportConfig
from ..dependencies.redis_connection import init_redis
from ..schemas.report import ReportCropped
from ..schemas.other import SuccessRequest
from ..schemas.log import log_transaction, StatusTraining

router = APIRouter()
ROOT_DIR = os.path.abspath('.')
ROOT_DIR = os.path.join(ROOT_DIR, 'assets')


@router.get('/get-models', response_model=SuccessRequest)
async def get_models_from_directory(background_task: BackgroundTasks):
    dir_models = os.path.join(ROOT_DIR, 'trained_models')

    background_task.add_task(
        log_transaction,
        method='/GET',
        endpoint='/get-models',
    )
    return {'status': True, 'detail': {'list_model': os.listdir(dir_models)}}


@router.get(
    '/crop-train',
    response_model=StatusTraining,
    status_code=status.HTTP_200_OK,
)
async def report_training_status(
        background_task: BackgroundTasks,
        uuid: str,
        redis_conn=Depends(init_redis)
):
    redis_model = await redis_conn.get(uuid)
    if redis_model is None:
        raise HTTPException(status_code=status.HTTP_200_OK,
                            detail='Not in progress train model.')
    redis_model = json.loads(redis_model)

    background_task.add_task(
        log_transaction,
        method='/GET',
        endpoint='/report/training',
    )
    return redis_model


@router.post('/cropped/complete', response_model=ReportCropped)
async def return_error_cropped_image(config: ReportConfig, background_task: BackgroundTasks):
    try:
        details_missing = []
        datasets = os.path.join(ROOT_DIR, config.datasets_folder)
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

    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/cropped/complete',
    )
    return model_store
