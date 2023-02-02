import shutil
import os
from fastapi import APIRouter, status, BackgroundTasks
from ..schemas.log import log_transaction
from ..engine import crop_extraction_faces, Config, train_model_facials
from ..dependencies import DATA_CACHE

router = APIRouter()


@router.post(
    '/cropped',
    status_code=status.HTTP_200_OK,
)
async def execute_cropped_preprocessing(background_task: BackgroundTasks):
    """
    Execute Crop preprocessing APIs\n
    **Warning please check configuration path directory feature image valid you can check "{host}/setting/config"**
    ### You can check status progress at "{host}/report/training"

    """

    model_store = Config(**DATA_CACHE)
    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/executor/cropped',
    )
    background_task.add_task(
        crop_extraction_faces,
        model_store
    )
    return {'status': True, 'detail': 'Please waiting for progress crop image.'}


@router.post('/training/model')
async def execute_training_model(background_task: BackgroundTasks):
    """
    Execute Training model APIs\n
    **Warning please check configuration path directory feature image valid you can check "{host}/setting/config"**
    ### You can check status progress at "{host}/report/training"

    """

    model_store = Config(**DATA_CACHE)
    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/executor/training/model',
    )
    background_task.add_task(
        train_model_facials,
        model_store
    )
    return {'status': True, 'detail': 'Please waiting for progress training model.'}


@router.post('/cropped/delete/labels')
async def cropped_delete_folders_label(config: Config):
    root_dir = os.path.abspath('.')
    datasets = os.path.join(root_dir, config.datasets_folder)
    for data in os.listdir(datasets):
        if not data.startswith('.'):
            path_ids = os.path.join(datasets, data)
            path_label = os.path.join(path_ids, config.labels_folder)
            if os.path.exists(path_label):
                shutil.rmtree(path_label)
    return config
