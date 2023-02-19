import json
import os
import time
import cv2
import pickle
import dlib
from .model import Config
from ..schemas.log import LOGGER
from ..dependencies.redis_connection import init_redis

FACE_DETECTOR = []
FACE_NAME = []
time_avg = []

PATH_ROOT_MODEL = os.path.abspath('server/engine/models')
sp_68_face = os.path.join(PATH_ROOT_MODEL, 'shape_predictor_68_face_landmarks.dat')
fr_model_v1 = os.path.join(PATH_ROOT_MODEL, 'dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(sp_68_face)
model = dlib.face_recognition_model_v1(fr_model_v1)


async def train_model_facials(config: Config, uuid: str):
    """
    Training dlib model

    :param uuid:
    :param config:
    :return:
    """
    LOGGER.info(f'execute payload: {config.dict() if config.dict() else "error payload"}')
    redis_conn = await init_redis()
    report_progress = {
        'training_method': 'Training model progressing',
        'training_status_success': False,
        'training_status': '0.0 %'
    }
    await redis_conn.set(uuid, json.dumps(report_progress))

    stff = time.time()
    datasets_folder = os.listdir(config.datasets_folder)
    for length in range(len(datasets_folder)):
        training_progress = round((length / len(datasets_folder)) * 100, 2)
        if not datasets_folder[length].startswith('.'):
            path_ids = os.path.join(config.datasets_folder, datasets_folder[length])
            path_labels = os.path.join(path_ids, config.labels_folder)
            for face_cropped in os.listdir(path_labels):
                path_facial = os.path.join(path_labels, face_cropped)
                img = cv2.imread(path_facial, cv2.COLOR_BGR2RGB)
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    shape = sp(img, d)
                    face_desc = model.compute_face_descriptor(img, shape, 1)
                    FACE_DETECTOR.append(face_desc)
                    FACE_NAME.append(datasets_folder[length])
                    sec = (time.time() - stff)
                    time_avg.append(sec)
            report_progress['training_progress'] = f'{training_progress} %'
            await redis_conn.set(uuid, json.dumps(report_progress))
            LOGGER.info(
                f'check labels while train in folder: {datasets_folder[length]} each label: {os.listdir(path_labels)}')
            LOGGER.info(f'progress trained is {training_progress} %')

    """
        Evaluate and save model in directory {config.file_trained_model}
    """

    avg = time_avg[-1] / len(time_avg)
    LOGGER.info('time spending: {} sec '.format(str(round(avg, 2))))
    root_dir = os.path.abspath('.')
    root_dir = os.path.join(root_dir, 'assets')
    path_trained_model = os.path.join(root_dir, 'trained_models')
    save_model_dir = os.path.join(path_trained_model, config.model_file)
    pickle.dump((FACE_DETECTOR, FACE_NAME), open(f'{save_model_dir}.pk', 'wb'))
    LOGGER.info(f'progress trained is 100 %')
    report_progress['training_method'] = 'Trained model is complete'
    report_progress['training_status_success'] = True
    report_progress['training_progress'] = '100 %'
    await redis_conn.set(uuid, json.dumps(report_progress))
    LOGGER.info('=======Finish trained model========')
