import os
import aiofiles
from fastapi import APIRouter, UploadFile, Depends, File, HTTPException, status, BackgroundTasks
from ..engine import predict
from ..schemas.predict import SelectModel, Predicted
from ..schemas.log import log_transaction

router = APIRouter()
ROOT_DIR = os.path.abspath('.')
PATH_REGEX = r'^(?![0-9._/])(?!.*[._]$)(?!.*\d_)(?!.*_\d)[a-zA-Z0-9_/]+$'


async def remove_img_cache(img):
    os.remove(img)


@router.post('/', response_model=Predicted)
async def prediction_image(
        background_task: BackgroundTasks,
        config: SelectModel = Depends(),
        file: UploadFile = File(...),
):
    allowed_extensions = ("jpg", "jpeg", "png")

    static_dir = os.path.join(ROOT_DIR, 'server/static')
    save_img = os.path.join(static_dir, 'save_cache_img')
    ext = file.filename.split(".")[-1]
    if ext not in allowed_extensions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File type not supported")

    img = os.path.join(save_img, file.filename)
    async with aiofiles.open(img, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    predicted = await predict(img=img, model_file=config.model_file)

    background_task.add_task(
        log_transaction,
        method='/POST',
        endpoint='/predict/',
    )
    background_task.add_task(
        remove_img_cache,
        img
    )
    return Predicted(**predicted)
