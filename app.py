from __future__ import annotations
import os
import random
import uuid
from datetime import datetime
import gradio as gr
import numpy as np
from PIL import Image
import torch
from diffusers import ConsistencyDecoderVAE, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from sa_solver_diffusers import SASolverScheduler
from transformers import T5EncoderModel
import time  # Import the time module

# Import argparse for command line argument parsing
import argparse

custom_css = """
img.data-testid {
    width: 1024px;
    height: 1024px;
}
"""

# Define command line arguments
parser = argparse.ArgumentParser(description="Gradio App with 8bit and 512model options")

# Rename the arguments to valid Python identifiers
parser.add_argument("--use_8bit", action="store_true", help="Use 8bit option")
parser.add_argument("--use_512model", action="store_true", help="Use 512model option")
parser.add_argument("--use_DallE_VAE", action="store_true", help="Use 512model option")
parser.add_argument("--share", action="store_true", help="Generates public Gradio Link")

args = parser.parse_args()

# Access the arguments correctly
use_8bit = args.use_8bit
use_512model = args.use_512model
use_DallE_VAE = args.use_DallE_VAE
use_Share = args.share

DESCRIPTION = """Original Source https://pixart-alpha.github.io/ \n
			This APP is modified and brought you by SECourses : https://www.patreon.com/SECourses
        """
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU �� This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["DPM-Solver", "SA-Solver"]
DEFAULT_SCHEDULE_NAME = "DPM-Solver"
NUM_IMAGES_PER_PROMPT = 1

if use_512model:
    model_path = "PixArt-alpha/PixArt-XL-2-512x512"
else:
    model_path = "PixArt-alpha/PixArt-XL-2-1024-MS"
	
	
if use_512model:
    use_res = 512
else:
    use_res = 1024	


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


if torch.cuda.is_available():
    if use_8bit:
        text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            load_in_8bit=True,
            device_map="auto",
        )
        pipe = PixArtAlphaPipeline.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if use_DallE_VAE:
            print("Using DALL-E 3 Consistency Decoder")
            pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16, device_map="auto")
        # speed-up T5
    else:
        pipe = PixArtAlphaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        print("Loaded on Device!")

    if use_DallE_VAE:
        print("Using DALL-E 3 Consistency Decoder")
        pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16, device_map="auto")
        # speed-up T5

    pipe.text_encoder.to_bettertransformer()

    if USE_TORCH_COMPILE:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")


def create_output_folders():
    base_dir = "outputs"
    today = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(base_dir, today)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Modified save_image function
def save_image(img):
    folder_path = create_output_folders()
    unique_name = str(uuid.uuid4()) + ".png"
    file_path = os.path.join(folder_path, unique_name)
    img.save(file_path)
    return file_path

# Modified randomize_seed_fn function
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

# Modified generate function to include batch count
def generate(
    prompt: str,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = use_res,
    height: int = use_res,
    schedule: str = DEFAULT_SCHEDULE_NAME,
    dpms_guidance_scale: float = 4.5,
    sas_guidance_scale: float = 3,
    dpms_inference_steps: int = 20,
    sas_inference_steps: int = 25,
    randomize_seed: bool = False,
    batch_count: str = "1",
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    image_paths = []
    
    batch_count_int = int(batch_count)
    counter = 0
    base_prompt = prompt
    base_neg_prompt = negative_prompt
    print(f"Starting generating {batch_count_int} images ...")
    for _ in range(batch_count_int):
        start_time = time.time()
        seed = int(randomize_seed_fn(seed, randomize_seed))
        generator = torch.Generator().manual_seed(seed)
        counter=counter+1
        if schedule == 'DPM-Solver':
            if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
                pipe.scheduler = DPMSolverMultistepScheduler()
            num_inference_steps = dpms_inference_steps
            guidance_scale = dpms_guidance_scale
        elif schedule == "SA-Solver":
            if not isinstance(pipe.scheduler, SASolverScheduler):
                pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction', tau_func=lambda t: 1 if 200 <= t <= 800 else 0, predictor_order=2, corrector_order=2)
            num_inference_steps = sas_inference_steps
            guidance_scale = sas_guidance_scale
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        if not use_negative_prompt:
            negative_prompt = None  # type: ignore
        prompt, negative_prompt = apply_style(style, base_prompt, base_neg_prompt)

        images = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
            use_resolution_binning=use_resolution_binning,
            output_type="pil",
        ).images
        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        avg_step_duration_ms = int((duration / num_inference_steps) * 1000)  # Calculate average step duration in milliseconds
        print(f"Image {counter} / {batch_count} - avg image duration {duration:.2f} seconds - step duration {avg_step_duration_ms} ms")  # Print the duration
        image_paths.extend([save_image(img) for img in images])

    return image_paths, seed


examples = [
    "A"
]
gallery_size=1
with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=10,
                placeholder="Enter your prompt",
                container=True,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, show_label=False, scale=2,allow_preview=True,preview=True,object_fit="scale-down")
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
        schedule = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=SCHEDULE_NAME,
            value=DEFAULT_SCHEDULE_NAME,
            label="Sampler Schedule",
            visible=True,
        )
        style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label="Image Style",
        )
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=False,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        batch_count = gr.Text(label="Batch count", value="1")
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=use_res,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=use_res,
            )
        with gr.Row():
            dpms_guidance_scale = gr.Slider(
                label="DPM-Solver Guidance scale",
                minimum=1,
                maximum=10,
                step=0.1,
                value=4.5,
            )
            dpms_inference_steps = gr.Slider(
                label="DPM-Solver inference steps",
                minimum=5,
                maximum=100,
                step=1,
                value=20,
            )
        with gr.Row():
            sas_guidance_scale = gr.Slider(
                label="SA-Solver Guidance scale",
                minimum=1,
                maximum=10,
                step=0.1,
                value=3,
            )
            sas_inference_steps = gr.Slider(
                label="SA-Solver inference steps",
                minimum=10,
                maximum=100,
                step=1,
                value=25,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            seed,
            width,
            height,
            schedule,
            dpms_guidance_scale,
            sas_guidance_scale,
            dpms_inference_steps,
            sas_inference_steps,
            randomize_seed,
            batch_count,  # Added batch_count
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=use_Share, inbrowser=True)

