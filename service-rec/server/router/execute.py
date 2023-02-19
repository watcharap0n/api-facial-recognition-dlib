import shutil
import os
from uuid import uuid4
from fastapi import APIRouter, status, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from ..schemas.log import log_transaction
from ..schemas.other import SuccessRequest
from ..engine import (
    crop_extraction_faces,
    Config,
    ConfigTrain,
    train_model_facials,
    ReportConfig,
    ConfigCrop
)

router = APIRouter()
ROOT_DIR = os.path.abspath('.')
ROOT_DIR = os.path.join(ROOT_DIR, 'assets')

responses_delete_labels = {
    400: {'description': 'File not found error please check your path directory'},
}

responses_async = {
    400: {'description': 'File not found error please check your path directory'},
}


@router.post(
    '/cropped',
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SuccessRequest,
    responses=responses_async
)
async def execute_cropped_preprocessing(config: ConfigCrop, background_task: BackgroundTasks):
    """
    Execute Crop preprocessing APIs\n
    ** **Warning please check configuration path directory feature image is correct you can check "/setting/config"**
    ### You can check status progress in "/report/training"

    """
    dataset_folder = os.path.join(ROOT_DIR, config.datasets_folder)
    if len(os.listdir(dataset_folder)) <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Your dataset is too small, can't be used for preprocessing.")

    uid = uuid4().hex
    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/executor/cropped',
    )
    background_task.add_task(
        crop_extraction_faces,
        config,
        uid
    )
    return {'status': True, 'detail': {
        'message': 'Please waiting for progress crop image.',
        'uuid': uid
    }}


@router.post(
    '/training/model',
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SuccessRequest,
    responses=responses_async
)
async def execute_training_model(config: ConfigTrain, background_task: BackgroundTasks):
    """
    Execute Training model APIs\n
    ** **Warning please check configuration path directory feature image is correct you can check "/setting/config"**
    ### You can check status progress at "/report/training"

    """

    dataset_folder = os.path.join(ROOT_DIR, config.datasets_folder)
    if not os.path.exists(dataset_folder):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Please check your folder for the training model or '
                   'before training you can check in api /report/cropped'
        )

    id_folder = os.listdir(dataset_folder)[0]
    full_id_folder = os.path.join(dataset_folder, id_folder)

    if not os.path.exists(os.path.join(full_id_folder, config.labels_folder)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Please check your folder for the training model or '
                   'before training you can check in api /report/cropped'
        )
    uid = uuid4().hex

    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/executor/training/model',
    )
    background_task.add_task(
        train_model_facials,
        config,
        uid
    )
    return {'status': True, 'detail': {
        'message': 'Please waiting for progress training model.',
        'uuid': uid
    }}


@router.post(
    '/crop-train',
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SuccessRequest,
    responses=responses_async
)
async def crop_and_train_model(config: Config, background_task: BackgroundTasks):
    dataset_folder = os.path.join(ROOT_DIR, config.datasets_folder)
    if len(os.listdir(dataset_folder)) <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Your dataset is too small, can't be used for preprocessing.")
    uid = uuid4().hex

    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/executor/crop-train',
    )
    background_task.add_task(
        crop_extraction_faces,
        config,
        uid,
        True
    )
    return {'status': True, 'detail': {
        'message': 'Please waiting for progress crop image.',
        'uuid': uid
    }}


@router.post('/cropped/delete/labels', response_model=Config, responses=responses_delete_labels)
async def cropped_delete_folders_label(config: ReportConfig, background_task: BackgroundTasks):
    datasets = os.path.join(ROOT_DIR, config.datasets_folder)
    for data in os.listdir(datasets):
        if not data.startswith('.'):
            path_ids = os.path.join(datasets, data)
            path_label = os.path.join(path_ids, config.labels_folder)
            if os.path.exists(path_label):
                shutil.rmtree(path_label)
    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/executor/cropped/delete/labels',
    )
    return config


@router.delete('/model/delete/{file}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(file: str, background_task: BackgroundTasks):
    folder_model = os.path.join(ROOT_DIR, 'trained_models')
    if not file.endswith('.pk'):
        join_pk = f'{file}.pk'
        file_model = os.path.join(join_pk, join_pk)
        os.remove(file_model)
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content='Deleted model file is success.')
    file_model = os.path.join(folder_model, file)
    os.remove(file_model)
    background_task.add_task(
        log_transaction,
        method='/DELETE',
        endpoint=f'/executor/model/delete/{file}',
    )
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content='Deleted model file is success.')
