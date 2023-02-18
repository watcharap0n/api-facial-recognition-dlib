import shutil
import os
from fastapi import APIRouter, status, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from ..schemas.log import log_transaction
from ..schemas.other import SuccessRequest
from ..dependencies import DATA_CACHE
from ..engine import crop_extraction_faces, Config, train_model_facials, ReportConfig

router = APIRouter()

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
async def execute_cropped_preprocessing(background_task: BackgroundTasks):
    """
    Execute Crop preprocessing APIs\n
    ** **Warning please check configuration path directory feature image is correct you can check "/setting/config"**
    ### You can check status progress in "/report/training"

    """
    model_store = Config(**DATA_CACHE)
    root_dir = os.path.abspath('.')
    dataset_folder = os.path.join(root_dir, model_store.datasets_folder)
    if len(os.listdir(dataset_folder)) <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Your dataset is too small, can't be used for preprocessing.")

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

    model_store = Config(**DATA_CACHE)
    root_dir = os.path.abspath('.')
    dataset_folder = os.path.join(root_dir, model_store.datasets_folder)
    id_folder = os.listdir(dataset_folder)[0]
    full_id_folder = os.path.join(dataset_folder, id_folder)
    if not os.path.exists(os.path.join(full_id_folder, model_store.labels_folder)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Please check your folder for the training model or '
                   'before training you can check in api /report/cropped'
        )

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


@router.post('/cropped/delete/labels', response_model=Config, responses=responses_delete_labels)
async def cropped_delete_folders_label(config: ReportConfig):
    root_dir = os.path.abspath('.')
    datasets = os.path.join(root_dir, config.datasets_folder)
    for data in os.listdir(datasets):
        if not data.startswith('.'):
            path_ids = os.path.join(datasets, data)
            path_label = os.path.join(path_ids, config.labels_folder)
            if os.path.exists(path_label):
                shutil.rmtree(path_label)
    return config


@router.delete('/model/delete/{file}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(file: str):
    root_dir = os.path.abspath('.')
    folder_model = os.path.join(root_dir, file)
    if not file.endswith('.pk'):
        join_pk = f'{file}.pk'
        file_model = os.path.join(join_pk, join_pk)
        os.remove(file_model)
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content='Deleted model file is success.')
    file_model = os.path.join(folder_model, file)
