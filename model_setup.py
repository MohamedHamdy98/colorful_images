from pathlib import Path
import subprocess

def setup_model_and_images():
    # Create directories for models and images using Path
    models_dir = Path("./models")
    images_dir = Path("./images")

    models_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download the model using aria2c
    print("Downloading model...")
    subprocess.run([
        "aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M",
        "https://huggingface.co/camenduru/cv_ddcolor_image-colorization/resolve/main/pytorch_model.pt",
        "-d", str(models_dir), "-o", "pytorch_model.pt"
    ], check=True)

    # Download the sample image
    print("Downloading sample image...")
    subprocess.run([
        "wget", "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/audrey_hepburn.jpg",
        "-O", str(images_dir / "input.jpg")
    ], check=True)

    print("Model and images setup completed!")

if __name__ == "__main__":
    setup_model_and_images()
