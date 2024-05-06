from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, redirect, url_for

import torch
from torchvision import transforms
# from diffusers import StableDiffusionPipeline
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# # Load model
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
# pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


# Load the SAM model and processor
try:
    print("----------model1-------------")
    model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
    print("----------modal cuda-------------")
    model.to("cuda")
    print("----------modal2-------------")
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    print("SAM model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading SAM model and processor: {e}")



@app.route('/')
def index():
  try:
    print("Initial page loaded!")
    img = load_image("https://picsum.photos/seed/picsum/200/300")
    print("Image generated! Converting image ...", img)

    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    img_bytes = base64.b64encode(img_bytes)
    img_bytes = img_bytes.decode("utf-8")
    print("Image converted! Sending image ...")


    input_points = [[[320, 600]]] # input point for object selection

    inputs = processor(img, input_points=input_points, return_tensors="pt").to("cuda")
    print("Inputs generated! Running model ...")
    outputs = model(**inputs)
    print("Model run successfully! Post processing masks ...")
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    print("Masks post processed! Getting number of mask images ...")



    # get number of mask images
    len(masks[0][0])
    print("Number of mask images:(1): ", len(masks[0][0]))


    print("input_points_2")
    input_points_2 = [[[200, 850]]]
    inputs_2 = processor(img, input_points=input_points_2, return_tensors="pt").to("cuda")
    outputs_2 = model(**inputs_2)
    masks_2 = processor.image_processor.post_process_masks(outputs_2.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    len(masks_2[0][0])
    print("Number of mask images(2): ", len(masks_2[0][0]))



    # Create a ToPILImage transform
    to_pil = transforms.ToPILImage()

    # Convert boolean tensors to binary tensors
    binary_matrix_1 = masks[0][0][2].to(dtype=torch.uint8)
    # binary_matrix_2 = masks_2[0][0][1].to(dtype=torch.uint8)

    print("binary_matrix_1: ", binary_matrix_1)
    # print("binary_matrix_2: ", binary_matrix_2)

    # apply the transform to the tensors
    mask_1 = to_pil(binary_matrix_1*255)
    # mask_2 = to_pil(binary_matrix_2*255)

    print("mask_1: ", mask_1)
    # print("mask_2: ", mask_2)

    # display original image with masks
    # make_image_grid([img, mask_1, mask_2], cols = 3, rows = 1)
    grid = make_image_grid([img, mask_1], cols = 2, rows = 1)

    print("Original image with masks displayed!")

    img_grid = make_image_grid([img, mask_1, mask_2], cols = 3, rows = 1)
    # img_grid to png
    img_grid_bytes = BytesIO()
    img_grid.save(img_grid_bytes, format="PNG")
    img_grid_bytes = img_grid_bytes.getvalue()
    img_grid_bytes = base64.b64encode(img_grid_bytes)
    img_grid_bytes = img_grid_bytes.decode("utf-8")



  # create inpainting pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "redstonehero/ReV_Animated_Inpainting",
        torch_dtype=torch.float16
    )
    print("Inpainting pipeline created!")
    pipeline.enable_model_cpu_offload()
    print("Model offload enabled!")
   

    # inpaint the image
    prompt = """flower-print, t-shirt """

    # generate image
    print("Generating image ...")
    img = pipeline.inpaint(
        prompt=prompt,
        width=512,
        height=768,
        num_inference_steps=24,
        image=img,
        mask_image=mask_1,
        guidance_scale=3,
        strength=1.0
      ).images[0]
    print("Image generated! Converting image ...", img)

    # display input image and generated image
    finalImg = make_image_grid([img.resize([512,768]), image], rows = 1, cols = 2)

    print("Input image and generated image displayed!",finalImg)

    # convert image to bytes
    # img_final_bytes = BytesIO()
    # finalImg.save(img_final_bytes, format="PNG")
    # img_final_bytes = img_final_bytes.getvalue()
    # img_final_bytes = base64.b64encode(img_final_bytes)
    # img_final_bytes = img_final_bytes.decode("utf-8")
    # print("Image converted! Sending image ...")



    
  # convert grid to bytes
    grid_image = BytesIO()
    grid.save(grid_image, format="PNG")
    grid_image = grid_image.getvalue()
    grid_image = base64.b64encode(grid_image)
    grid_image = grid_image.decode("utf-8")
    print("Grid image converted! Sending image ...")

    


    return render_template('index.html', img_bytes=grid_image)
  except Exception as e:
    print(f"Error loading initial---: {e}")
    return "Error loading initial page.======>"

if __name__ == '__main__':
    app.run()