import threestudio
import torch
import torch.nn.functional as F
from threestudio.models.guidance.stable_zero123_guidance import StableZero123Guidance
from threestudio.utils.typing import *
from tqdm import tqdm


@threestudio.register("mvimg-gen-stable-zero123-guidance")
class StableZero123GuidanceInference(StableZero123Guidance):
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, cond, latents_noisy, noise_pred):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(30)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents_noisy.shape[0]
        idxs = torch.zeros(bs, dtype=torch.long, device=self.device)

        # fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        # imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_noisy[b : b + 1]
            c = {
                "c_crossattn": [cond["c_crossattn"][0][[b, b + len(idxs)], ...]],
                "c_concat": [cond["c_concat"][0][[b, b + len(idxs)], ...]],
            }
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                x_in = torch.cat([latents] * 2)
                t_in = torch.cat([t.reshape(1)] * 2).to(self.device)
                noise_pred = self.model.apply_model(x_in, t_in, c)
                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "imgs_final": imgs_final,
        }

    def __call__(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        **kwargs,
    ):
        batch_size = elevation.shape[0]
        latents: Float[Tensor, "B 4 64 64"]
        latents = torch.zeros((batch_size, 4, 64, 64), device=self.device)

        cond = self.get_cond(elevation, azimuth, camera_distances)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.scheduler.num_train_timesteps - 1,
            self.scheduler.num_train_timesteps,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

        guidance_out = self.guidance_eval(cond, latents_noisy, noise)

        return guidance_out
