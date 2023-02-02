import os
import time
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from .schemas.log import LOGGER
from .router import setting, report, executor

app = FastAPI(
    version=os.environ.get('SERVER_VERSION', '1.0.1'),
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json',
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
Face Recognition. 👋🏻

## APIs

You can **read items each API**.

You will be able to:

***prefix /***
"""


@app.get('/404', response_class=HTMLResponse)
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
        title="Face Recognition",
        version="1.0.1",
        description=description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = customer_openapi_signature


@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    process_time = '{:2f}'.format(process_time)
    response.headers['X-Process-Time'] = str(process_time)
    pass_url = str(request.url)
    sentence = '../../' or '..%2F..%2F' or '/../../'
    if sentence in pass_url:
        return RedirectResponse(status_code=status.HTTP_303_SEE_OTHER, url='/404')
    return response


app.include_router(
    setting.router,
    prefix='/setting',
    tags=['Setting']
)

app.include_router(
    report.router,
    prefix='/report',
    tags=['Report']
)

app.include_router(
    executor.router,
    prefix='/executor',
    tags=['Executor']
)


@app.on_event("startup")
async def startup_event():
    """Start up event for FastAPI application."""
    LOGGER.info("Starting up server signature")
    root_pj = os.path.dirname(__file__)
    static_dir = os.path.join(root_pj, 'static')
    os.makedirs(static_dir, exist_ok=True)
    root_dir = os.path.abspath('.')
    trained_folder = os.path.join(root_dir, 'trained_models')
    os.makedirs(trained_folder, exist_ok=True)

