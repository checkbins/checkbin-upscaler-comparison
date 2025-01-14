from modal import (
    Image as ModalImage,
    App,
    Mount,
    Volume,
    Secret,
    gpu,
    build,
    enter,
    method,
    asgi_app,
)
import os, sys
sys.path.insert(0, 'checkbin-python/src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkbin
from facefusion.processors.frame.core import load_frame_processor_module
from facefusion.processors.frame.modules.face_enhancer import apply_enhance, prepare_crop_frame, normalize_crop_frame
from facefusion.processors.frame import globals as frame_processors_globals
import requests
import numpy as np
from PIL import Image
from io import BytesIO

test_upscalers_image = (
    ModalImage.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg", "libsm6", "libxext6", "git"])
    .pip_install(
        [
            "torch==2.4.0", # This doesn't seem to work, torch is 2.1.2 when printed later? 
            "torchvision",
        ]
    )
    .pip_install(
        [
            "filetype==1.2.0",
            "gradio==3.50.2",
            "numpy==1.26.4",
            "onnx==1.16.0",
            "opencv-python==4.9.0.80",
            "psutil==5.9.8",
            "tqdm==4.66.4",
            "scipy==1.13.0",
            "firebase-admin",
            "azure-storage-blob",
            "boto3",
            "google-cloud-storage",
            "onnxruntime-gpu==1.17.1",
            # "diffusers",
            "transformers",
            "xformers",
            "accelerate",
            "torch",
            "torchvision",
            "safetensors",
            "spandrel",
        ]
    )
    .run_commands(
        "pip install git+https://github.com/huggingface/diffusers.git"
    )
    .pip_install(
        "huggingface-hub"
        
    )
    .run_commands(
        "pip install transformers[sentencepiece]"
    )
    #     ["pip show torch", "pip3 install natten==0.17.1+torch210cu121 -f https://shi-labs.com/natten/wheels/"]
    # )
)
app = App("alias-playground", image=test_upscalers_image)

frame_processors_globals.face_enhancer_model = 'gfpgan_1.4'

def run_classic_face_enhance(crop_vision_frame):  
    crop_vision_frame = np.array(Image.fromarray(crop_vision_frame).resize((512, 512)))
    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    crop_vision_frame = apply_enhance(crop_vision_frame)
    return normalize_crop_frame(crop_vision_frame)


def run_flux_controlnet_upscaler(control_image): 
    import torch
    from diffusers.utils import load_image
    from diffusers.models import FluxControlNetModel
    from diffusers.pipelines import FluxControlNetPipeline
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    
    # Load pipeline
    controlnet = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    # Load a control image
    # control_image = load_image(
    #     "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/examples/input.jpg"
    # )
    w, h = control_image.size

    # Upscale x4
    control_image = control_image.resize((w * 4, h * 4))

    image = pipe(
        prompt="", 
        control_image=control_image,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=28, 
        guidance_scale=3.5,
        height=control_image.size[1],
        width=control_image.size[0]
    ).images[0]
    return image

def run_4xFaceUpDATUpscaler(crop_vision_frame_tensor):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    model = ModelLoader().load_from_file(r"local/models/4xFaceUpDAT.pth")  
    assert isinstance(model, ImageModelDescriptor)

    model.cuda().eval()
    return model(crop_vision_frame_tensor)
    

def run_4xFaceUpDATSharpUpscaler(crop_vision_frame_tensor):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    model = ModelLoader().load_from_file(r"local/models/4xFaceUpSharpDAT.pth")  
    assert isinstance(model, ImageModelDescriptor)

    model.cuda().eval()
    return model(crop_vision_frame_tensor)

@app.function(
    gpu=gpu.A100(size="80GB"),
    timeout=3600,
    image=test_upscalers_image,
    mounts=[Mount.from_local_dir("..", remote_path="/root/"), Mount.from_local_dir("models", remote_path="/root/local/models")]
    # mounts=[Mount.from_local_dir("controlnet_depth", remote_path="/root/controlnet_depth"), Mount.from_local_dir("./own_controlnet", remote_path="/root/own_controlnet")]
)
def run_batch():
    import torch
    face_enhancer = load_frame_processor_module("face_enhancer")
    checkbin.authenticate(token=os.environ["CHECKBIN_TOKEN"])
    # checkbin_app = checkbin.CheckbinApp(app_key="alias_face_swap", mode="remote")
    # runners = checkbin_app.start_run(set_id="88e0b4df-4f47-4e00-8418-098a05a45fdb") # These are upscaler test examples!
    checkbin_app = checkbin.CheckbinApp(app_key="upscaler_demo", mode="remote")
    runners = checkbin_app.start_run(set_id="6b39c90c-807d-4e0d-8714-d30e62a74b86") # These are the 100x celeb faces!
    checkbin_app.add_azure_credentials(
        account_name=os.environ["AZURE_ACCOUNT_NAME"],
        account_key=os.environ["AZURE_ACCOUNT_KEY"],
    )

    for runner in runners:
        # This the key in the alias test set
        # crop_vision_frame_url = runner.input_state['swapped_face_before_paste']['url']
        for key in runner.input_state.keys():
            print(key)
        for key in runner.input_state['image_128'].keys():
            print(key)
        crop_vision_frame_url = runner.input_state['image_128']['url']
        print(crop_vision_frame_url)

        response = requests.get(crop_vision_frame_url)
        crop_vision_frame_url_image = Image.open(BytesIO(response.content))
        crop_vision_frame = np.array(crop_vision_frame_url_image)

        crop_vision_frame_tensor = torch.tensor(crop_vision_frame.copy())
        crop_vision_frame_tensor = crop_vision_frame_tensor / 255.0
        crop_vision_frame_tensor = crop_vision_frame_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        crop_vision_frame_tensor = crop_vision_frame_tensor.to("cuda")

        # Checkpoint! 
        runner.checkpoint("4xFaceUpDAT Upscaler")
        upscaled_image = run_4xFaceUpDATUpscaler(crop_vision_frame_tensor)
        upscaled_image = upscaled_image.permute(0, 2, 3, 1).squeeze(0)
        upscaled_image_numpy = upscaled_image.cpu().numpy()
        upscaled_image_numpy_int8 = (upscaled_image_numpy * 255).astype(np.uint8)
        upscaled_image = Image.fromarray(upscaled_image_numpy_int8)
        upscaled_image.save("upscaled_image.jpg")
        runner.upload_file("checkbin", "azure", "4xFaceUpDAT_upscaled", "upscaled_image.jpg", "image")
        os.remove("upscaled_image.jpg") 

        runner.checkpoint("4xFaceUpDATSharp Upscaler")
        upscaled_image = run_4xFaceUpDATSharpUpscaler(crop_vision_frame_tensor)
        upscaled_image = upscaled_image.permute(0, 2, 3, 1).squeeze(0)
        upscaled_image_numpy = upscaled_image.cpu().numpy()
        upscaled_image_numpy_int8 = (upscaled_image_numpy * 255).astype(np.uint8)
        upscaled_image = Image.fromarray(upscaled_image_numpy_int8)
        upscaled_image.save("upscaled_image.jpg")
        runner.upload_file("checkbin", "azure", "4xFaceUpDATSharp_upscaled", "upscaled_image.jpg", "image")
        os.remove("upscaled_image.jpg") 

        flux_upscaled = run_flux_controlnet_upscaler(crop_vision_frame_url_image)
        runner.checkpoint("Flux ControlNet Upscaler")
        flux_upscaled.save("flux_upscaled.jpg")
        runner.upload_file("checkbin", "azure", "flux_controlnet_upscaled", "flux_upscaled.jpg", "image")

        crop_vision_frame_enhanced = run_classic_face_enhance(crop_vision_frame=crop_vision_frame.copy())
        runner.checkpoint("GFPGan Enhance")
        image = Image.fromarray(crop_vision_frame_enhanced)
        image.save("crop_vision_frame_enhanced.jpg")
        runner.upload_file("checkbin", "azure", "gfpgan_enhanced", "crop_vision_frame_enhanced.jpg", "image")
        os.remove("crop_vision_frame_enhanced.jpg")
        
        runner.submit_test()

    print("Checkbin Run Id: " + str(runners[0].run_id))

    # crop_vision_frame = runner.sta
