from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, redirect, url_for

import rembg
import torch
import numpy as np
from PIL import Image,ImageOps
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import matplotlib.pyplot as plt
import base64
from io import BytesIO


# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


# Load the SAM model and processor
try:
    print(torch.__version__)
    print("----------model1-------------")
    # Create inpainting pipeline
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "redstonehero/ReV_Animated_Inpainting", 
        torch_dtype=torch.float16
    )
    print("Inpainting pipeline created!")
    pipeline.enable_model_cpu_offload()
    print("Model offload enabled!")
except Exception as e:
    print(f"Error loading SAM model and processor: {e}")



@app.route('/')
def index():
  try:
    print("Initial page loaded!")
    
    # Load the input image
    init_image = load_image("https://picsum.photos/id/870/512/768").resize((512,768))

    # Convert the input image to a numpy array
    input_array = np.array(init_image)

    # Extract mask using rembg
    mask_array = rembg.remove(input_array, only_mask=True)

    # Create a PIL Image from the output array
    mask_image = Image.fromarray(mask_array)

    print("Image generated! Converting image ...", img)

    # convert image to bytes
    img_out = BytesIO()
    init_image.save(img_out, format="PNG")
    img_out = img_out.getvalue()
    img_out = base64.b64encode(img_out)
    img_out = img_out.decode("utf-8")
    print("Image converted! Sending image ...")


    mask_image_inverted = ImageOps.invert(mask_image)
    

    # convert PIL image to bytes
    mask_1 = BytesIO()
    mask_image_inverted.save(mask_1, format="PNG")
    mask_1 = mask_1.getvalue()
    mask_1 = base64.b64encode(mask_1)
    mask_1 = mask_1.decode("utf-8")
    print("Mask image converted! Sending image ...")




    prompt = """night sky with a moon and saturn"""
 
    negative_prompt = "tail, deformed, mutated, ugly, disfigured"
 
    image = pipeline(prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=768,
        num_inference_steps=20,
        image=init_image, 
        mask_image=mask_image_inverted,
        guidance_scale=1,
        strength=0.7, 
        generator=torch.manual_seed(189018)
      ).images[0]
    print("Image generated! Converting image ...", image)

    # display input image and generated image
    finalImg = make_image_grid([init_image, image], rows=1, cols=2)

    print("Input image and generated image displayed!",finalImg)

    # convert image to bytes
    img_final_bytes = BytesIO()
    finalImg.save(img_final_bytes, format="PNG")
    img_final_bytes = img_final_bytes.getvalue()
    img_final_bytes = base64.b64encode(img_final_bytes)
    img_final_bytes = img_final_bytes.decode("utf-8")
    print("Image converted! Sending image ...")


    


    return render_template('index.html', img_bytes=finalImg)
  except Exception as e:
    print(f"Error loading initial---: {e}")
    return "Error loading initial page.======>"

if __name__ == '__main__':
    app.run()