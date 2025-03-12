import torch
from diffusers import AutoencoderKL, DDIMScheduler
from .latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from typing import Callable, List, Optional, Union
from .latentsync.utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from .latentsync.utils.image_processor import ImageProcessor


from pathlib import Path
from omegaconf import OmegaConf
from .latentsync.whisper.audio2feature import Audio2Feature
from diffusers.utils.import_utils import is_xformers_available
from .latentsync.models.unet import UNet3DConditionModel
from accelerate.utils import set_seed
import time
import numpy as np

CONFIG_PATH = Path("latentsync/configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("latentsync/checkpoints/latentsync_unet.pt")

def load_all_model():
    config = OmegaConf.load(CONFIG_PATH)
    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    scheduler = DDIMScheduler.from_pretrained("latentsync/configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "latentsync/checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "latentsync/checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")
    
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        CHECKPOINT_PATH.absolute().as_posix(),  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = Pipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")
    seed = -1
    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    return audio_encoder,vae,unet,pipeline

def _load_avatar(video_path,
                   pipeline):
        # Create the temp directory if it doesn't exist
        output_dir = Path("./temp")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert paths to absolute Path objects and normalize them
        video_file_path = Path(video_path)
        video_path = video_file_path.absolute().as_posix()
        
        print(f"Input video path: {video_path}")

        
        faces, original_video_frames, boxes, affine_matrices = pipeline.affine_transform_video(video_path)
        return faces, original_video_frames, boxes, affine_matrices 


class Pipeline(LipsyncPipeline):  
    def __init__(self, vae, audio_encoder, unet, scheduler):
        super().__init__(vae, audio_encoder, unet, scheduler)
        self.height, self.width = None, None

    def warm_up(self,
                height: Optional[int] = None,
                width: Optional[int] = None,
                mask: str = "fix_mask",
                num_frames: int = 16,
                callback_steps: Optional[int] = 1,):
        
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

         # 1. Default height and width to unet
        self.height = height or self.unet.config.sample_size * self.vae_scale_factor
        self.width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

    @torch.no_grad()
    def inference(
        self,
        whisper_chunks,
        faces, original_video_frames, boxes, affine_matrices,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        num_inference_steps: int = 2,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        start_time = time.perf_counter()
        self.unet.eval()
        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        # if self.unet.add_audio_layer:
        #     whisper_feature = self.audio_encoder.audio2feat(audio_path)
        #     whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        #     num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
        # else:
        #     num_inferences = len(faces) // num_frames

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            num_frames * 1,
            num_channels_latents,
            self.height,
            self.width,
            weight_dtype,
            device,
            generator,
        )


        if self.unet.add_audio_layer:
            audio_embeds = torch.stack(whisper_chunks)
            audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
            if do_classifier_free_guidance:
                null_audio_embeds = torch.zeros_like(audio_embeds)
                audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
        else: 
            audio_embeds = None
            
        inference_faces = faces
        latents = all_latents
        pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
            inference_faces, affine_transform=False
        )

        # 7. Prepare mask latent variables
        mask_latents, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_pixel_values,
            self.height,
            self.width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Prepare image latents
        image_latents = self.prepare_image_latents(
            pixel_values,
            device,
            weight_dtype,
            generator,
            do_classifier_free_guidance,
        )

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        end_time = time.perf_counter()
        print(f"推理预处理执行时间: {end_time - start_time:.6f} 秒")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for j, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat(
                    [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                )

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and j % callback_steps == 0:
                        callback(j, t, latents)

        start_time = time.perf_counter()
        print(f"推理过程执行时间: {start_time - end_time:.6f} 秒")
        # Recover the pixel values
        decoded_latents = self.decode_latents(latents)
        decoded_latents = self.paste_surrounding_pixels_back(
            decoded_latents, pixel_values, 1 - masks, device, weight_dtype
        )
        synced_video_frames.append(decoded_latents)
        # masked_video_frames.append(masked_pixel_values)

        end_time = time.perf_counter()
        print(f"推理后处理前半段执行时间: {end_time - start_time:.6f} 秒")

        # synced_video_frames = self.restore_video(
        #     torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        # )
        # masked_video_frames = self.restore_video(
        #     torch.cat(masked_video_frames), original_video_frames, boxes, affine_matrices
        # )

        end_time = time.perf_counter()
        print(f"推理后处理后半段执行时间: {end_time - start_time:.6f} 秒")
        # return synced_video_frames
        return np.stack(synced_video_frames, axis=0) 