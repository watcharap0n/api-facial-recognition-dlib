import os
import cv2
import dlib
import pickle
import time
import uuid
import numpy as np
from ..schemas.log import LOGGER

PATH_ROOT_MODEL = os.path.abspath('server/engine/models')
sp_68_face = os.path.join(PATH_ROOT_MODEL, 'shape_predictor_68_face_landmarks.dat')
fr_model_v1 = os.path.join(PATH_ROOT_MODEL, 'dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(sp_68_face)
model = dlib.face_recognition_model_v1(fr_model_v1)


async def predict(img: str, model_file: str):
    """

    :param img:
    :param file_trained_model:
    :return:
    """

    NAMES = []
    UNKNOWN = []
    FACE_DESC, FACE_NAME = pickle.load(open(model_file, 'rb'))

    image = cv2.imread(img)
    height, width, color = image.shape
    scale = 0.5
    if height >= 1000:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        LOGGER.info('resizing...')

    time_start = time.time()
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected = detector(gray_scale, 1)
    for det in detected:
        xy = det.left(), det.top()
        wh = det.right(), det.bottom()
        shape = sp(gray_scale, det)
        face_desc_first = model.compute_face_descriptor(image, shape, 1)

        det = []
        for face_desc in FACE_DESC:
            det.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc_first)))
        det = np.array(det)
        idx = np.argmin(det)
        if det[idx] <= 0.34:
            name = FACE_NAME[idx]
            name = str(name)
            LOGGER.info(f'Precision {det[idx]} {name}')
            percent = 1.0 - det[idx]
            percent = percent * 100
            cv2.putText(image, f'{name}', (xy[0], xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f'{round(percent, 1)}%', (xy[0] + 40, xy[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(image, xy, wh, (0, 255, 0), 2)
            NAMES.append(name)
        elif det[idx] <= 0.39:
            name = FACE_NAME[idx]
            name = str(name)
            LOGGER.info(f'Less Precision {det[idx]} {name}')
            percent = 1.0 - det[idx]
            percent = percent * 100
            cv2.putText(image, f'{name}', (xy[0], xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(image, f'{round(percent, 1)}%', (xy[0] + 40, xy[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, xy, wh, (0, 255, 255), 2)
            NAMES.append(name)
        else:
            possible_person = FACE_NAME[idx]
            possible_person = str(possible_person)
            name = 'unknown'
            UNKNOWN.append(name)
            percent = 1.0 - det[idx]
            percent = percent * 100
            cv2.putText(image, possible_person, (xy[0], xy[1] + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(image, f'{round(percent, 1)}%', (xy[0], xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.rectangle(image, xy, wh, (0, 0, 255), 2)

    img_name = uuid.uuid4().hex
    ROOT_DIR = os.path.abspath('.')
    pred_dir = os.path.join(ROOT_DIR, 'server/static/images/predicted')
    save_img = os.path.join(pred_dir, img_name)
    cv2.imwrite(filename=f'{save_img}.png', img=image)
    endtime = time.time() - time_start
    LOGGER.info(f'time spend: {round(endtime, 2)}')
    LOGGER.info(NAMES)

    result = {
        'recognized': NAMES,
        'unknowns': len(UNKNOWN),
        'peoples': len(NAMES) + len(UNKNOWN),
        'id_img': img_name,
        'spend_time': endtime
    }
    return result
