import os
import time
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from .schemas.log import LOGGER
from .router import report, execute, predict, image

app = FastAPI(
    version=os.environ.get('SERVER_VERSION', '0.0.1'),
    docs_url='/api/v1/docs',
    redoc_url='/api/v1/redoc',
    openapi_url='/api/v1/openapi.json',
    include_in_schema=os.getenv('OPENAPI_SCHEMA', True),
)

script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, 'static/')
os.makedirs(st_abs_file_path, exist_ok=True)
app.mount('/static', StaticFiles(directory=st_abs_file_path), name='static')

origins = [
    "http://localhost:80",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

description = """
### APIs AI/ML (DLIB shape 68 marks) 

service available to use management structure model machine learning such as preprocessing myself and training model myself all service with APIs you can read documentation below.

---

**Documentation:** <a href="https://fastapi.tiangolo.com" target="_blank">https://fastapi.tiangolo.com</a>

**Developed @MangoConsultant**

---
"""


@app.get('/404', response_class=HTMLResponse, tags=['others'])
async def not_found_page():
    return """
        <html>
            <head>
                <title> Not found in here </title>
            <head>
            <body>
                <h1> Not found page </h1>
            </body>
        </html>
        """


def customer_openapi_signature():
    """
    docs description API
    :return:
        -> func
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Face Recognition Variables API",
        version="0.0.1",
        description=description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = customer_openapi_signature

app.include_router(
    report.router,
    prefix='/report',
    tags=['reports']
)

app.include_router(
    execute.router,
    prefix='/execute',
    tags=['executes'],
)

app.include_router(
    predict.router,
    prefix='/predict',
    tags=['predictions']
)

app.include_router(
    image.router,
    prefix='/image',
    tags=['images']
)


@app.on_event("startup")
async def startup_event():
    """Start up event for FastAPI application."""
    LOGGER.info("Starting up server signature")
    root_pj = os.path.dirname(__file__)
    static_dir = os.path.join(root_pj, 'static')
    os.makedirs(static_dir, exist_ok=True)
    root_dir = os.path.abspath('.')
    assets_folder = os.path.join(root_dir, 'assets')
    os.makedirs(assets_folder, exist_ok=True)
    trained_folder = os.path.join(root_dir, 'assets/trained_models')
    os.makedirs(trained_folder, exist_ok=True)
