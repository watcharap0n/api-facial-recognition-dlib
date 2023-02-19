import os
from fastapi import APIRouter, HTTPException, status, Path, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from ..schemas.other import SuccessRequest
from ..schemas.log import log_transaction

router = APIRouter()
ROOT_DIR = os.path.abspath('.')

responses_file = {
    400: {'description': 'File not found error please check your path directory'},
}


@router.get('/get-id', response_model=SuccessRequest)
async def get_id_images(background_task: BackgroundTasks):
    pred_dir = os.path.join(ROOT_DIR, 'server/static/images/predicted')

    background_task.add_task(
        log_transaction,
        method='/GET',
        endpoint=f'/get-id',
    )
    return {'status': True, 'detail': {
        'id_images': os.listdir(pred_dir)
    }}


@router.get(
    '/{id_img}', response_class=FileResponse,
    responses=responses_file
)
async def download_image_predicted(
        background_task: BackgroundTasks,
        id_img: str = Path(description='ID Image on folder directory static/images/predicted/')
):
    if not id_img.endswith('.png'):
        id_img = f'{id_img}.png'
    pred_dir = os.path.join(ROOT_DIR, 'server/static/images/predicted')
    path_img = os.path.join(pred_dir, f'{id_img}')
    if not os.path.exists(path_img):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Not found image {id_img}')

    background_task.add_task(
        log_transaction,
        method='/GET',
        endpoint=f'/image/{id_img}',
    )
    return FileResponse(path=path_img, filename=os.path.basename(path_img), media_type='image/png')


@router.delete('/{id_img}', status_code=status.HTTP_204_NO_CONTENT)
async def remove_image(
        background_task: BackgroundTasks,
        id_img: str = Path(description='ID Image on folder directory static/images/predicted/')
):
    pred_dir = os.path.join(ROOT_DIR, 'server/static/images/predicted')
    path_img = os.path.join(pred_dir, f'{id_img}')
    if not os.path.exists(path_img):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Not found image {id_img}')
    os.remove(path_img)

    background_task.add_task(
        log_transaction,
        method='/DELETE',
        endpoint=f'/image/{id_img}',
    )
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content='Deleted model file is success.')
