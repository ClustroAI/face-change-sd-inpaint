import json
import requests
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torchvision.transforms as transforms 

from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

torch.backends.cuda.matmul.allow_tf32 = True

# SD inpainting
pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
        )
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                    use_safetensors=True,
                                    torch_dtype=torch.float16
                                    )

pipe.vae = vae
pipe.enable_xformers_memory_efficient_attention()
#pipe.vae.enable_tiling()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.scheduler.config.algorithm_type = 'sde-dpmsolver++'
pipe.safety_checker = None
pipe.to("cuda")

# clothes segmentation
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to("cuda")

def scale_image(img, max_size=1500):
    # Open the image

    # Get the current size of the image
    width, height = img.size

    # Calculate the scaling factor for both height and width
    scale_factor = 1 if width <= max_size and height <= max_size else min(max_size / width, max_size / height)

    # Calculate the new dimensions for the scaled image (multiples of 8)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Ensure the new dimensions are multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

def face_segmentation(image):
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    outputs = model(**inputs)

    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    ''' 
    Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
            5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 
            11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 
            16: "Bag", 17: "Scarf"
    '''
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = (pred_seg == 11)
    transform = transforms.ToPILImage()
    mask = transform(mask * 1.0)

    # add mask blur
    mask = mask.filter(ImageFilter.GaussianBlur(radius = 15))
    pixel_map = mask.load()
    w,h = mask.size
    for i in range(w):
        for j in range(h):
            pixel_map[i,j] = 0 if pixel_map[i,j] < 110 else 255
            
    return mask

def invoke(input_text):
    input_json = json.loads(input_text)
    image_url = input_json['image_url']
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = scale_image(image)
    mask_image = face_segmentation(image)

    prompt = input_json['prompt']
    prompt = f'{prompt}, (portrait), (best quality:1.4), (ultra highres:1.2), (photorealistic:1.4), (soft focus:1.4)'
    if 'negative_prompt' in input_json:
        negative_prompt = input_json['negative_prompt']
    else:
        negative_prompt = '''poorly rendered face, poorly rendered eyes, deformed face, deformed eyelid, 
                            collapsed eyeshadow, multiple eyebrows,inaccurate eyes, showing teeth, long neck, 
                            (cross eye), bad proportions, gross proportions,(ugly face),mask,out of frame, closed eyes
                            '''
    
    strength = float(input_json['strength']) if 'strength' in input_json else 1.0
    num_inference_steps = int(input_json['steps']) if 'steps' in input_json else 30
    guidance_scale = float(input_json['guidance_scale']) if 'guidance_scale' in input_json else 6.5
    width, height = image.size

    result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
    ).images[0]
    result.save("generated_image.png")

    return "generated_image.png"
