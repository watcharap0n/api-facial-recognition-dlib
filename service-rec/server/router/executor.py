import shutil
import os
from fastapi import APIRouter, status, BackgroundTasks, HTTPException
from ..schemas.log import log_transaction
from ..schemas.other import SuccessRequest
from ..dependencies import DATA_CACHE
from ..engine import crop_extraction_faces, Config, train_model_facials

router = APIRouter()

responses_delete_labels = {
    404: {'description': 'File not found error please check your path directory'},
}

responses_async = {
    500: {'description': 'Something wrong please try again later'}
}


@router.post(
    '/cropped',
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SuccessRequest,
    responses=responses_async
)
async def execute_cropped_preprocessing(background_task: BackgroundTasks):
    """
    Execute Crop preprocessing APIs\n
    ** **Warning please check configuration path directory feature image is correct you can check "/setting/config"**
    ### You can check status progress in "/report/training"

    """
    try:
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
        return {'status': True, 'detail': {
            'message': 'Please waiting for progress crop image.'
        }}

    except Exception as exe:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f'Something wrong {exe.args} please try again later.')


@router.post(
    '/training/model',
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SuccessRequest,
    responses=responses_async
)
async def execute_training_model(background_task: BackgroundTasks):
    """
    Execute Training model APIs\n
    ** **Warning please check configuration path directory feature image is correct you can check "/setting/config"**
    ### You can check status progress at "/report/training"

    """

    try:
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
        return {'status': True, 'detail': {
            'message': 'Please waiting for progress training model.'
        }}

    except Exception as exe:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f'Something wrong {exe.args} please try again later.')


@router.post('/cropped/delete/labels', response_model=Config, responses=responses_delete_labels)
async def cropped_delete_folders_label(config: Config):
    root_dir = os.path.abspath('.')
    datasets = os.path.join(root_dir, config.datasets_folder)
    folder_model = os.path.join(root_dir, 'trained_models')
    file_model = os.path.join(folder_model, config.file_trained_model)
    try:
        os.remove(file_model)
    except FileNotFoundError as exe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'File not found error {exe.filename}')

    for data in os.listdir(datasets):
        if not data.startswith('.'):
            path_ids = os.path.join(datasets, data)
            path_label = os.path.join(path_ids, config.labels_folder)
            if os.path.exists(path_label):
                shutil.rmtree(path_label)
    return config
