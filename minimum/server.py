# server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from io import BytesIO
from PIL import Image, ImageDraw
import time, random, os

app = FastAPI()

@app.get("/")
def serve_index():
    # Serve index.html from the same folder as this file
    return FileResponse("index.html")

@app.get("/generate")
def generate():
    # Simulate slow image generation
    time.sleep(2 + random.random() * 2)
    img = Image.new("RGB", (400, 200), (220, 240, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 80), "Generated image", fill=(0, 0, 0))

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
