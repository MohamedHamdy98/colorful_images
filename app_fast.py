from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import cv2
from image_colorization import ImageColorizationPipeline
from model_setup import setup_model_and_images
import uvicorn

app = FastAPI()

# Ensure model and images are downloaded and setup
setup_model_and_images()

# Initialize the colorizer with the pre-trained model
colorizer = ImageColorizationPipeline(model_path="./models/pytorch_model.pt", input_size=512)

@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...)):
    input_image_path = f"./images/{file.filename}"
    output_image_path = f"./images/output_{file.filename}"

    # Save the uploaded file
    with open(input_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image using OpenCV
    img = cv2.imread(input_image_path)

    # Colorize the image
    result_img = colorizer.process(img)

    # Save the result image
    cv2.imwrite(output_image_path, result_img)

    return FileResponse(output_image_path)

@app.get("/")
def home():
    return {"message": "Welcome to the Image Colorization API!"}

if __name__ == "__main__":
    uvicorn.run("app_fast:app", host="0.0.0.0", port=5005, reload=True)