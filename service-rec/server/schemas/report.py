from typing import Optional, Union, List
from pydantic import BaseModel


class Detail(BaseModel):
    img_original: Optional[int] = None
    cropped_success: Optional[int] = None
    id_folder: Union[str, None] = None
    percentage: Union[str, None] = None
    error_images: Union[List, None] = []

    class Config:
        validate_assigment = True
        schema_extra = {
            'example': {
                'img_original': 10,
                'cropped_success': 8,
                'id_folder': 'user01',
                'percentage': '80.00 %',
                'error_images': [],
            }
        }


class ReportCropped(BaseModel):
    status: Optional[bool] = False
    detail: Union[List[Detail], None] = []

    class Config:
        schema_extra = {
            'example': {
                'status': False,
                'detail': []
            }
        }
