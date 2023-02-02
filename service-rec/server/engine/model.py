import pytz
import os
import re
from datetime import datetime
from typing import Union, Optional
from pydantic import BaseModel, validator, Field

PATH_REGEX = r'^(?![0-9._/])(?!.*[._]$)(?!.*\d_)(?!.*_\d)[a-zA-Z0-9_/]+$'
ROOT_DIR = os.path.abspath('.')


class Config(BaseModel):
    datasets_folder: Union[str, None] = os.path.join(ROOT_DIR, 'datasets')
    labels_folder: Union[str, None] = 'labels'
    file_trained_model: Union[str, None] = 'trained_model'
    scale: Optional[float] = 0.5
    img_pixel: Optional[int] = 128
    recent_time: Optional[datetime] = None

    @validator('datasets_folder', pre=True)
    def validate_datasets_folder(cls, value):
        value_path = os.path.join(ROOT_DIR, value)
        os.makedirs(value_path, exist_ok=True)
        split_current = os.path.split(value_path)[1]
        if len(split_current) > 20:
            raise ValueError('Name cannot be longer than 20 characters')
        if not re.match(PATH_REGEX, split_current):
            raise ValueError('Invalid path directory format.')
        return value_path

    @validator('labels_folder', pre=True)
    def validate_labels_folder(cls, value):
        if len(value) > 20:
            raise ValueError('Name cannot be longer than 20 characters')
        if not re.match(PATH_REGEX, value):
            raise ValueError('Invalid path directory format.')
        return value

    @validator('recent_time', pre=True, always=True)
    def set_name(cls, value):
        tz = pytz.timezone('Asia/Bangkok')
        dt = datetime.now(tz)
        return dt

    class Config:
        validate_assignment = True
        schema_extra = {
            'example': {
                'datasets_folder': 'datasets',
                'labels_folder': 'labels',
                'file_trained_model': 'trained_model',
                'scale': 0.5,
                'img_pixel': 128,
            }
        }


class ReportConfig(BaseModel):
    datasets_folder: Union[str, None] = os.path.join(ROOT_DIR, 'datasets')
    labels_folder: Union[str, None] = 'labels'
    file_trained_model: Union[str, None] = 'trained_model'
    scale: Optional[float] = 0.5
    img_pixel: Optional[int] = 128
    recent_time: Optional[datetime] = None

    @validator('datasets_folder', pre=True)
    def validate_datasets_folder(cls, value):
        value_path = os.path.join(ROOT_DIR, value)
        split_current = os.path.split(value_path)[1]
        if len(split_current) > 20:
            raise ValueError('Name cannot be longer than 20 characters')
        if not re.match(PATH_REGEX, split_current):
            raise ValueError('Invalid path directory format.')
        return value_path

    @validator('labels_folder', pre=True)
    def validate_labels_folder(cls, value):
        if len(value) > 20:
            raise ValueError('Name cannot be longer than 20 characters')
        if not re.match(PATH_REGEX, value):
            raise ValueError('Invalid path directory format.')
        return value

    @validator('recent_time', pre=True, always=True)
    def set_name(cls, value):
        tz = pytz.timezone('Asia/Bangkok')
        dt = datetime.now(tz)
        return dt

    class Config:
        validate_assignment = True
        schema_extra = {
            'example': {
                'datasets_folder': 'datasets',
                'labels_folder': 'labels',
                'file_trained_model': 'trained_model',
                'scale': 0.5,
                'img_pixel': 128,
            }
        }
