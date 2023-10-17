import io

import PIL
import numpy as np
import uvicorn
from PIL import Image
from typing import List
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils import scale_image, pil_to_b64, build_hist

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('main.html', {'request': request})


@app.post("/")
async def make_image(request: Request, scale: int = Form(), files: List[UploadFile] = File(description='UploadFile')):
    ready = False
    if len(files) > 0 and len(files[0].filename) > 0:
        ready = True

    request_result = []

    if ready:
        content = [await file.read() for file in files]
        img_data = {}
        for idx, con in enumerate(content):
            try:
                source_image = Image.open(io.BytesIO(con)).convert('RGB')
                scaled_image = scale_image(source_image, scale)
                img_data['source_image'] = pil_to_b64(source_image)
                img_data['scaled_image'] = pil_to_b64(scaled_image)
                img_data['source_hist'] = build_hist(np.array(source_image))
                img_data['scaled_hist'] = build_hist(np.array(scaled_image))
                img_data['valid_image'] = True
            except PIL.UnidentifiedImageError:
                img_data['valid_image'] = False
            finally:
                img_data['filename'] = files[idx].filename
                request_result.append(img_data)

    return templates.TemplateResponse(
        'main.html',
        {
            'request': request,
            'ready': ready,
            'request_result': request_result
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
