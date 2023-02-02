import os
import json
import dlib
import cv2
from .model import Config
from ..schemas.log import LOGGER
from ..dependencies.redis_connection import init_redis

PATH_ROOT_MODEL = 'models'
detector = dlib.get_frontal_face_detector()


async def crop_extraction_faces(config: Config):
    """ Crop facials to directory each path_labels

    :param config:
    :param detector:

    :return:
    """
    LOGGER.info(f'execute payload: {config.dict() if config.dict() else "error payload"}')
    redis_conn = await init_redis()
    report_progress = {
        'training_method': 'Crop extraction progressing',
        'training_status_success': False,
    }

    datasets_folder = os.listdir(config.datasets_folder)
    for ids in range(len(datasets_folder)):
        cropped_progress = round((ids / len(datasets_folder)) * 100, 2)
        if not datasets_folder[ids].startswith('.'):
            path_ids = os.path.join(config.datasets_folder, datasets_folder[ids])
            path_labels = os.path.join(path_ids, config.labels_folder)
            if not path_labels.startswith('.'):
                os.makedirs(path_labels, exist_ok=True)
            for i in os.listdir(path_ids):
                path_img = os.path.join(path_ids, i)
                if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg') or i.endswith('.JPG'):
                    image_color = cv2.imread(path_img)
                    image = cv2.resize(
                        image_color,
                        None,
                        fx=config.scale,
                        fy=config.scale,
                        interpolation=cv2.INTER_AREA
                    )
                    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dets = detector(gray_scale, 1)
                    for k, d in enumerate(dets):
                        x, y = d.left(), d.top()
                        w, h = d.right(), d.bottom()
                        cropped_image = image[y:h, x:w]  # cropped_image
                        image = cv2.resize(
                            cropped_image, (
                                config.img_pixel,
                                config.img_pixel
                            ), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(path_labels, i), image)

            report_progress['training_progress'] = f'{cropped_progress} %'
            await redis_conn.set('training_model', json.dumps(report_progress))
            LOGGER.info(f'check labels in folder: {datasets_folder[ids]} each label: {os.listdir(path_labels)}')
            LOGGER.info(f'progress copped is {cropped_progress} %')

    LOGGER.info(f'progress copped is 100 %')
    report_progress['training_method'] = 'Cropped images is complete'
    report_progress['training_status_success'] = False
    report_progress['training_progress'] = '100 %'
    await redis_conn.set('training_model', json.dumps(report_progress))
    LOGGER.info('=======Finish cropped========')
