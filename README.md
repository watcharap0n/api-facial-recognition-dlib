# Facial Recognition APIs Model ML (DLIB shape 68 marks) 

service available to use management structure model machine learning such as preprocessing myself and training model myself all service with APIs you can read documentation below.

---

**Documentation:** <a href="https://fastapi.tiangolo.com" target="_blank">https://fastapi.tiangolo.com</a>

**Source code:** <a href="https://github.com/watcharap0n/api-facial-recognition-dlib" target="_blank">https://github.com/watcharap0n/api-facial-recognition-dlib</a>

---


## Requirements

Python 3.8 ++

You can install dependencies package in directory source code **requirements.txt**

## Installation

```bash
$ pip install virsualenv 
$ virsualenv venv
$ source venv/bin/activate

(venv) $ pip install -r service-rec/requirements.txt
(venv) $ cd service-rec
(venv) $ uvicorn server.main:app --reload --port 8080 --host 0.0.0.0
 
```

### Before execution

first before execution path prefix /executor. you can setting directory put image to directory for training model & preprocessing 

- API  **/setting/config**
  - payload body
      ```python
      class Config(BaseModel):
          datasets_folder: Union[str, None] = os.path.join(ROOT_DIR, 'datasets')
          labels_folder: Union[str, None] = 'labels'
          file_trained_model: Union[str, None] = 'trained_model'
          scale: Optional[float] = 0.5
          img_pixel: Optional[int] = 128
          recent_time: Optional[datetime] = None
      ```
    
**You can check status between training APIs**

- API  **/report/training**  check status training model
- API **/report/cropped** check status crop preprocessing before put to model dlib
- API **/report/delete/labels** for delete images cropped in directory folder





