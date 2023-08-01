#predict.py

from cog import BasePredictor, Input, Path
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import numpy as np
from typing import List
# from utils import download_model
from gradio_lineart import process
from gradio_normalbae import process as normalbae_process
from gradio_mlsd import process as mlsd_process

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # download_model("https://huggingface.co/SG161222/Realistic_Vision_V2.0/resolve/main/Realistic_Vision_V2.0.ckpt", "./models")
        # download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth", "./models")
        
        # Load the state dictionary from Realistic_Vision_V2.0.ckpt
        state_dict = load_state_dict('./models/Realistic_Vision_V2.0.ckpt', location='cuda')

        # LINEART
        model_name = 'control_v11p_sd15_lineart'
        self.model = create_model(f'./models/{model_name}.yaml').cuda()
        self.model.load_state_dict(state_dict, strict=False)

        # NORMAL
        self.normal_model_name = "control_v11p_sd15_normalbae"
        self.normal_model = create_model(f'./models/{self.normal_model_name}.yaml').cuda()
        self.normal_model.load_state_dict(state_dict, strict=False)
        
        # MLSD
        self.MLSD_model_name = "control_v11p_sd15_mlsd"
        self.MLSD_model = create_model(f'./models/{self.MLSD_model_name}.yaml').cuda()
        self.MLSD_model.load_state_dict(state_dict, strict=False)

        # Continue with the rest of your code...
        self.ddim_sampler = DDIMSampler(self.model)
        self.normal_ddim_sampler = DDIMSampler(self.normal_model)
        self.MLSD_ddim_sampler = DDIMSampler(self.MLSD_model)
        

    def predict(
        self,
        image: Path = Input(
                    description="Input image"
                ),
                prompt: str = Input(
                    description="Prompt for the model",
                    default='Animal'
                ),
                structure: str = Input(
                    description="Structure to condition on",
                    choices=['lineart', 'normal', 'MLSD'],
                    default='lineart'
                ),
                num_samples: str = Input(
                    description="Number of samples (higher values may OOM)",
                    choices=['1', '4'],
                    default='1'
                ),
                image_resolution: int = Input(
                    description="Resolution of image",
                    default=512
                ),
                ddim_steps: int = Input(
                    description="Steps",
                    default=10
                ),
                strength: float = Input(
                    description="Control strength",
                    default=1.0
                ),
                scale: float = Input(
                    description="Scale for classifier-free guidance",
                    default=9.0,
                    ge=0.1,
                    le=30.0
                ),
                seed: int = Input(
                    description="Seed",
                    default=None
                ),
                eta: float = Input(
                    description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
                    default=0.0
                ),
                preprocessor: str= Input(
                    description="preprocessor, Normal bae just for Normal, Lineart and lineart coarse just for lineart, MLSD just for MLSD",
                    default="Lineart", 
                    choices=["Lineart", "Lineart_Coarse", "Normal_BAE","MLSD", "None"] 
                ),
                preprocessor_resolution: int = Input(
                    description="Preprocessor resolution",
                    default=512
                ),
                a_prompt: str = Input(
                    description="Additional text to be appended to prompt",
                    default="Best quality, extremely detailed"
                ),
                n_prompt: str = Input(
                    description="Negative prompt",
                    default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                ),
                guessmode: bool= Input(
                    description="Guess mode",
                    default=False
                                ),
                value_threshold: float = Input(
                    description="value_threshold, just for MLSD",
                    default=9.0,
                    ge=0.1,
                    le=30.0
                                ),
                distance_threshold: float = Input(
                    description="distance_threshold, JUST FOR MLSD",
                    default=9.0,
                    ge=0.1,
                    le=30.0
                ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        num_samples = int(num_samples)
        image_resolution = int(image_resolution)
        if not seed:
            seed = np.random.randint(1000000)
        else:
            seed = int(seed)

        # load input_image
        input_image = Image.open(image)
        # convert to numpy
        input_image = np.array(input_image)

        if structure == 'lineart':
            outputs = process(
                self.model,
                self.ddim_sampler,
                preprocessor,
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                preprocessor_resolution,
                ddim_steps,
                guessmode,
                strength,
                scale,
                seed,
                eta,
            )
        elif structure=='normal':
            outputs= normalbae_process(
                self.normal_model,
                self.normal_ddim_sampler,
                preprocessor,
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                preprocessor_resolution,
                ddim_steps,
                guessmode,
                strength,
                scale,
                seed,
                eta,
            )
        elif structure=='MLSD':
            outputs= mlsd_process(
                self.MLSD_model,
                self.MLSD_ddim_sampler,
                preprocessor,
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                preprocessor_resolution,
                ddim_steps,
                guessmode,
                strength,
                scale,
                seed,
                eta,
                value_threshold,
                distance_threshold
            )
        
        # outputs from list to PIL
        outputs = [Image.fromarray(output) for output in outputs]
        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        # return paths to output files
        return [Path(f"tmp/output_{i}.png") for i in range(len(outputs))]