import torch
from torch.cuda.amp import autocast



class DiffusionReconstructionModule:
    def __init__(self, pipe, lpips_loss_fn, class_center_buffer=None, device='cuda'):
        self.pipe = pipe
        self.vae_decoder_params = list(pipe.vae.decoder.parameters()) + \
                                  list(pipe.vae.post_quant_conv.parameters())
        self.lpips_loss_fn = lpips_loss_fn
        self.center_buffer = class_center_buffer
        self.device = device

    def reconstruct_from_latent(self, img, start_latents, num_inference_steps=50, strength=1, num_finetune_steps=15):

        pipe = self.pipe
        device = self.device

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        start_step = min(int(num_inference_steps * strength), num_inference_steps)
        timesteps = pipe.scheduler.timesteps[start_step:]

        latents = start_latents.clone().detach().requires_grad_(True)

        # with autocast():
        pixel_values = pipe.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
        image_embeddings = pipe.image_encoder(pixel_values).image_embeds[:, None, :]

        finetune_timesteps = timesteps[-num_finetune_steps:]

        for param in self.vae_decoder_params:
            param.requires_grad_(False) # Default to no gradients for VAE decoder

        for t in pipe.progress_bar(timesteps):
            # Check if current timestep is one of the fine-tuning steps
            is_finetune_step = t in finetune_timesteps
            
            # Conditionally enable/disable gradients for VAE decoder
            if is_finetune_step:
                for param in self.vae_decoder_params:
                    param.requires_grad_(True)
            else:
                for param in self.vae_decoder_params:
                    param.requires_grad_(False)

            # with autocast():
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=image_embeddings).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        for param in self.vae_decoder_params:
            param.requires_grad_(True) # Ensure it's trainable for the final decode from the loss

        # with autocast():
        latents_for_decode = 1 / pipe.vae.config.scaling_factor * latents
        recon_img = pipe.vae.decode(latents_for_decode).sample
        recon_img = (recon_img / 2 + 0.5).clamp(0, 1)

        return latents, recon_img

    def process_image(self, img, label, strength=0.92, alpha=0.3, mode='train', if_perb = True, num_finetune_steps=15):

        with torch.no_grad():#, autocast():
            initial_latents = self.pipe.vae.encode(img * 2 - 1).latent_dist.sample() * self.pipe.vae.config.scaling_factor
            
            noise = torch.randn_like(initial_latents)
            noisy_latents = self.pipe.scheduler.add_noise(initial_latents, noise, self.pipe.scheduler.timesteps[min(int(len(self.pipe.scheduler.timesteps) * strength), len(self.pipe.scheduler.timesteps)-1)])

        perturbed_latents = noisy_latents.clone()
        if self.center_buffer is not None:
            with torch.no_grad():#, autocast():
                '''
                direction = self.center_buffer.get_opposite_direction_no_label(perturbed_latents)
                latent_norm = torch.norm(perturbed_latents.flatten(1), dim=1).view(-1, 1, 1, 1)
                noise_norm = torch.norm(direction.flatten(1), dim=1).view(-1, 1, 1, 1)
                direction = direction / (noise_norm + 1e-8) * (latent_norm * alpha)
                perturbed_latents = perturbed_latents - direction
                '''
                direction = self.center_buffer.get_opposite_direction_no_label(perturbed_latents, alpha=alpha)
                if if_perb:
                    perturbed_latents = perturbed_latents - direction
        
        final_latents, recon_img = self.reconstruct_from_latent(
            img, perturbed_latents, strength=strength, num_finetune_steps=num_finetune_steps
        )

        return final_latents, recon_img