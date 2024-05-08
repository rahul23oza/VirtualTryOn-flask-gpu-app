
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
import os

import torch
from torchvision import transforms
from transformers import SamModel, SamProcessor
from diffusers import AutoPipelineForInpainting, StableDiffusionPipeline
from diffusers.utils import load_image, make_image_grid
import base64
from io import BytesIO

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def index():
    try:
        print("Initial page loaded!")
        init_img = load_image("https://livcasttestenvci.blob.core.windows.net/test/test2.jpg")
        print("Image Loaded! Converting image ...")

        img_bytes = BytesIO()
        init_img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        img_bytes = base64.b64encode(img_bytes)
        img_bytes = img_bytes.decode("utf-8")
        print("Image converted! Sending image ...")

        # Generate mask using SAM model
        print("----------SamModel-------------")
        model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
        print("----------SamModel to cuda-------------")
        model.to("cuda")
        print("----------SamProcessor-------------")
        processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
        print("SAM model and processor loaded successfully.")

        input_points = [[[320, 600]]]  # input point for object selection
        inputs = processor(init_img, input_points=input_points, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        print("Masks post processed! Getting number of mask images ...")

        print("Number of mask images:(1): ", len(masks[0][0]))

        # Create a ToPILImage transform
        to_pil = transforms.ToPILImage()

        # Convert boolean tensors to binary tensors
        binary_matrix_1 = masks[0][0][2].to(dtype=torch.uint8)
        # apply the transform to the tensors
        mask_1 = to_pil(binary_matrix_1 * 255)

        print("binary_matrix_1: ", binary_matrix_1)
        print("mask_1: ", mask_1)

        step_1 = make_image_grid([init_img, mask_1], cols = 2, rows = 1)
        print("Input image and mask displayed!", step_1)
        # Save the mask to a file
        mask_path = os.path.join('static', 'mask.png')
        mask_1.save(mask_path)

        step1_path = os.path.join('static', 'step1.png')
        step_1.save(step1_path)

        step1_bytes = BytesIO()
        step_1.save(step1_bytes, format="PNG")
        step1_bytes = step1_bytes.getvalue()
        step1_bytes = base64.b64encode(step1_bytes)
        step1_bytes = step1_bytes.decode("utf-8")
        print("Mask image converted! Sending image ...")


        # Release SAM model from memory
        del model, processor, inputs, outputs, masks

        # create inpainting pipeline
        # pipeline = StableDiffusionPipeline.from_pretrained(
        #     "redstonehero/ReV_Animated_Inpainting",
        #     torch_dtype=torch.float16,
        #     # use_safetensors=True,
        # )
        # print("Inpainting pipeline created!")
        # pipeline.enable_model_cpu_offload()

        pipeline = AutoPipelineForInpainting.from_pretrained(
            "redstonehero/ReV_Animated_Inpainting",
            torch_dtype=torch.float16
        )

        pipeline.enable_model_cpu_offload()

        print("Model offload enabled!")

        # Load the saved mask
        mask_1 = load_image(mask_path)

        # inpaint the image
        prompt = """flower-print, t-shirt """
        negative_prompt = " deformed, mutated, ugly, disfigured"
        # generate image
        print("Generating image ...")
        fin_image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=768,
            num_inference_steps=24,
            image=init_img,
            mask_image=mask_1,
            guidance_scale=3,
            strength=1.0
        ).images[0]

        # fin_image = pipeline.inpaint(
        #     prompt=prompt,
        #     image=init_img,
        #     mask_image=mask_1,
        #     height=768,
        #     width=512,
        #     num_inference_steps=24,
        #     guidance_scale=3,
        #     strength=1.0,
        # )["sample"][0]

        fin_bytes = BytesIO()
        fin_image.save(fin_bytes, format="PNG")
        fin_bytes = fin_bytes.getvalue()
        fin_bytes = base64.b64encode(fin_bytes)
        fin_bytes = fin_bytes.decode("utf-8")




        print("Image generated! Converting image ...",fin_image)

        # display input image and generated image
        finalImg = make_image_grid([fin_image.resize([512, 768]), fin_image], rows=1, cols=2)

        print("Input image and generated image displayed!", finalImg)

        # convert image to bytes
        img_final_bytes = BytesIO()
        finalImg.save(img_final_bytes, format="PNG")
        img_final_bytes = img_final_bytes.getvalue()
        img_final_bytes = base64.b64encode(img_final_bytes)
        img_final_bytes = img_final_bytes.decode("utf-8")
        print("Image converted! Sending image ...")

        return render_template(
            'index.html',
            img1=img_bytes, 
            step_1=step1_bytes,
            img2=fin_bytes, 
            img_bytes=img_final_bytes
        )
    except Exception as e:
        print(f"Error loading initial---: {e}")
        return "Error loading initial page.======>"

if __name__ == '__main__':
    app.run()