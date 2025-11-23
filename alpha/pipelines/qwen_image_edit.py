from dataclasses import dataclass

from torch import nn
from PIL import Image

# from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import *
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import *
# from diffusers import QwenImageEditPlusPipeline

@dataclass
class QwenImageEditModules:
    scheduler: FlowMatchEulerDiscreteScheduler
    vae: AutoencoderKLQwenImage
    text_encoder: Qwen2_5_VLForConditionalGeneration
    tokenizer: Qwen2Tokenizer
    processor: Qwen2VLProcessor
    transformer: QwenImageTransformer2DModel
    
    @classmethod
    def from_pipeline(cls, pipeline: QwenImageEditPlusPipeline):
        return cls(
            scheduler=pipeline.scheduler,
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            processor=pipeline.processor,
            transformer=pipeline.transformer,
        )
        
    def requires_grad_(self, requires_grad: bool):
        self.transformer.requires_grad_(requires_grad)
        self.text_encoder.requires_grad_(requires_grad)
        self.vae.requires_grad_(requires_grad)
    
    def to_dict(self):
        return {**self.__dict__}
    
    def to(self, *args, **kwargs):
        for module in self.to_dict().values():
            if isinstance(module, nn.Module):
                module.to(*args, **kwargs)




class CustomQwenImageEditPlusPipeline(QwenImageEditPlusPipeline):


    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
    def pack_latents_multi_frames(latents):
        """
        B, C, F, H, W -> B, F * H/2 * W/2, 4C
        """
        B, C, F, H, W = latents.shape
        latents = latents.view(B, C, F, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6) # B, F, H/2, W/2, C, 2, 2
        latents = latents.reshape(B, F * (H // 2) * (W // 2), C * 4) # B, F * H/2 * W/2, 4C
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._unpack_latents
    def unpack_latents_multi_frames(latents, frames, height_latents, width_latents):
        """
        B, F * H/2 * W/2, 4C -> B, C, F, H, W
        """
        B, L, C4 = latents.shape
        F = frames
        H = height_latents
        W = width_latents
        latents = latents.view(B, F, H // 2, W // 2, C4 // 4, 2, 2)
        latents = latents.permute(0, 4, 1, 2, 5, 3, 6) # B, C, F, H/2, 2, W/2, 2
        latents = latents.reshape(B, C4 // 4, F, H, W)
        return latents

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values if image else None,
            image_grid_thw=model_inputs.image_grid_thw if image else None,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask
    
    def prepare_condition_image(self, img: Image.Image, reshape: bool = True):
        if img.mode != "RGB":
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, (0, 0), img)
            img = bg
        image_width, image_height = img.size
        new_width, new_height = calculate_dimensions(
            CONDITION_IMAGE_SIZE if reshape else image_width * image_height, 
            image_width / image_height
        )
        return (
            self.image_processor.resize(img, new_width, new_height),
            (new_width, new_height)
        )
    
    def prepare_vae_image(self, img: Image.Image, reshape: bool = True):
        # RETURN: B=1 C F=1 H W
        image_width, image_height = img.size
        new_width, new_height = calculate_dimensions(
            VAE_IMAGE_SIZE if reshape else image_width * image_height, 
            image_width / image_height
        )
        return (
            self.image_processor.preprocess(img, new_width, new_height).unsqueeze(2),
            (new_width, new_height)
        )
        
    def prepare_images(
        self,
        images: Union[None, Image.Image, List[Image.Image]], 
        prepare_type: str,
        reshape: bool = True
    ):
        assert prepare_type in ['condition', 'vae'], "prepare_type must be either 'condition' or 'vae'"
        if images is None:
            return [], []
        
        shapes = []
        prepared_images = []
        
        if not isinstance(images, list):
            images = [images]
            
        for img in images:
            img, shape = {
                'condition': self.prepare_condition_image,
                'vae': self.prepare_vae_image
            }[prepare_type](img, reshape)
            prepared_images.append(img)
            shapes.append(shape)
        
        return prepared_images, shapes

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        frames=1,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image
                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    # expand init_latents for batch_size
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            shape = (batch_size, num_channels_latents, frames, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.pack_latents_multi_frames(latents)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        debug: bool = False,
        frames: int = 1,
        reshape: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
                Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
                equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
                enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
                encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
                lower image quality.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to None):
                A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
                where the guidance scale is applied during inference through noise prediction rescaling, guidance
                distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
                scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
                that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
                parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
                ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
                please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
                enable classifier-free guidance computations).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        if image is not None:
            one_image = image[0] if isinstance(image, list) else image
            # image_size = one_image.shape if isinstance(one_image, torch.Tensor) else one_image.size
            if isinstance(one_image, torch.Tensor):
                raise NotImplementedError("Pass PIL.Image.Image instead")
                # TODO: support tensor in the future versions
                input_width = one_image.shape[-1]
                input_height = one_image.shape[-2]
            else:
                input_width = one_image.size[0]
                input_height = one_image.size[1]
            calculated_width, calculated_height = calculate_dimensions(1024 * 1024, input_width / input_height)
            height = height or calculated_height
            width = width or calculated_width
        else:
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Preprocess 
        
        if debug:
            print("DEBUG 3")
            
        if (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            raise NotImplementedError
        
        has_condition = image is not None
            
        condition_images, condition_image_sizes = self.prepare_images(
            images=image,
            prepare_type='condition',
            reshape=reshape
        )
        
        vae_images, vae_image_sizes = self.prepare_images(
            images=image,
            prepare_type='vae',
            reshape=reshape
        )
        
        if debug:
            print(f"{vae_images[0].shape=}, {vae_image_sizes=}")
        
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        if debug:
            print(f"{prompt=}")
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images if has_condition else None,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        if debug:
            print(f"{condition_images=}")
            print(f"{prompt_embeds.shape=}, {prompt_embeds_mask.shape=}")
            print(f"{prompt_embeds.min()=}, {prompt_embeds.max()=}")
            print(f"{prompt_embeds.std()=}, {prompt_embeds.mean()=}")
        
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images if has_condition else None,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images if has_condition else None,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            frames,
        )
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        assert latents.shape[1] == frames * latent_height // 2 * latent_width // 2, "Latents shape is incorrect."
        img_shapes = [
            [
                *([(1, latent_height // 2, latent_width // 2)] * frames),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    # neg_noise_pred = neg_noise_pred[:, -latents.size(1):]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            if debug:
                print(f"{latents.shape=}, {frames=}, {latent_height=}, {latent_width=}")
            latents = self.unpack_latents_multi_frames(latents, frames, latent_height, latent_width) # B, C, F, H, W
            latents = latents.permute(0, 2, 1, 3, 4) # B, F, C, H, W
            B, F, C, H, W = latents.shape
            latents = latents.view(B * F, C, 1, H, W)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)
                

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)