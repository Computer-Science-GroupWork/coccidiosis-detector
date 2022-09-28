from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import sys, numpy as np

path = Path(__file__).parent
model_file_url = 'https://www.4sync.com/web/directDownload/PBlz3_Fa/f8Iqpe3-.0ec48e7e5a97cd110f45b835992abe82'
model_file_name = 'chickens'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FILE_SRC = '/tmp/saved_image.png'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    await download_file(model_file_url, MODEL_PATH)
    model = load_model(MODEL_PATH) # Load your Custom trained model
    model.make_predict_function()
    #model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/api/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    with open(IMG_FILE_SRC, 'wb') as f: f.write(img_bytes)
    return model_predict(IMG_FILE_SRC, model, bool_api=True)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    with open(IMG_FILE_SRC, 'wb') as f: f.write(img_bytes)
    return model_predict(IMG_FILE_SRC, model)

def model_predict(img_path, model, bool_api=False):
    img = image.load_img(img_path, target_size=(224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    prediction = model.predict(x)[0]
    cocci, healthy = prediction
    label = "Coccidiosis" if cocci > healthy else "Healthy"
    color = "red" if cocci > healthy else "green"
    accuracy = max(cocci, healthy)
    if not bool_api:
        result_html1 = path/'static'/'result1.html'
        result_html2 = path/'static'/'result2.html'
        result_html = str(result_html1.open().read() + '<span style=\"color: ' + color + ';\">' + label + '</span>' + ' at <span style=\"color: blue;\">' + str(round(accuracy*100, 2)) + '%</span> accuracy' + result_html2.open().read())
        return HTMLResponse(result_html)
    else:
        return JSONResponse({ 'label': label, 'accuracy': float(accuracy), 'code': 1 if cocci > healthy else 0})

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="127.0.0.1", port=8080)
