import os
from fastapi import status, HTTPException
from pydantic import BaseModel, validator
from typing import Union

ROOT_DIR = os.path.abspath('.')
ROOT_DIR = os.path.join(ROOT_DIR, 'assets')
PATH_REGEX = r'^(?![0-9._/])(?!.*[._]$)(?!.*\d_)(?!.*_\d)[a-zA-Z0-9_/]+$'


class Predicted(BaseModel):
    recognized: Union[list, None] = []
    unknowns: Union[int, None] = 0
    peoples: Union[int, None] = 0
    id_img: str
    spend_time: float


class SelectModel(BaseModel):
    model_file: str = 'example_model.pk'

    @validator('model_file', pre=True)
    def validate_trained_model_folder(cls, value):
        directory_path = os.path.join(ROOT_DIR, 'trained_models')
        file_path = os.path.join(directory_path, value)
        split_current = os.path.split(file_path)[1]
        if len(split_current) > 20:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail='Name cannot be longer than 20 characters')
        if not value.endswith('.pk'):
            join_pk = f'{file_path}.pk'
            if not os.path.isfile(join_pk):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f'Not found path file {join_pk}')
            return join_pk
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f'Not found path file {file_path}')
        return file_path

    class Config:
        schema_extra = {
            'example': {
                'model_file': 'example_model.pk'
            }
        }
