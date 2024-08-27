import torch
from comfy.k_diffusion.sampling import trange, to_d
import comfy.model_patcher
import comfy.samplers

@torch.no_grad()
def fast_distance_weights(t,p=2):
    d = torch.zeros_like(t,device=t.device)
    for i in range(t.shape[0]):
        d[i] = (t - t[i]).abs().sum(dim=0)
    d = (1 - (d - d.min()) / (d.max() - d.min())).pow(p)
    d = torch.nan_to_num(d,nan=1,neginf=1,posinf=1)
    d = (d / d.sum(dim=0))
    return (d * t).sum(dim=0)

# Euler and CFGpp part taken from comfy_extras/nodes_advanced_samplers
def distance_wrap(resample,resample_end=-1,cfgpp=False):
    @torch.no_grad()
    def sample_distance_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        if cfgpp:
            uncond = None
            def post_cfg_function(args):
                nonlocal uncond
                uncond = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function)

        s_min, s_max = sigmas[sigmas > 0].min(), sigmas.max()
        progression = lambda x: max(0,min(1,((x - s_min) / (s_max - s_min)) ** 0.5))

        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            sigma_hat = sigmas[i]
            denoised = model(x, sigma_hat * s_in, **extra_args)

            if cfgpp and torch.any(uncond):
                d = to_d(x - denoised + uncond, sigmas[i], denoised)
            else:
                d = to_d(x, sigma_hat, denoised)

            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat

            if resample_end >= 0:
                res_mul = progression(sigma_hat)
                resample_steps = max(min(resample,resample_end),min(max(resample,resample_end),int(resample * res_mul + resample_end * (1 - res_mul))))
            else:
                resample_steps = resample

            if sigmas[i + 1] == 0 or resample_steps == 0:
                # Euler method
                x = x + d * dt
            else:
                x_n = [d]
                for re_step in range(resample_steps):
                    x_new = x + d * dt
                    new_denoised = model(x_new, sigmas[i + 1] * s_in, **extra_args)
                    new_d = to_d(x_new, sigmas[i + 1], new_denoised)
                    x_n.append(new_d)
                    if re_step == 0:
                        d = (new_d + d) / 2
                    else:
                        d = fast_distance_weights(torch.stack(x_n), re_step + 2)
                x = x + d * dt
        return x
    return sample_distance_advanced

class SamplerDistanceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"resample": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1,
                                                  "tooltip":"0 all along gives Euler. 1 gives Heun.\nAnything starting from 2 will use the distance method."}),
                             "resample_end": ("INT", {"default": -1, "min": -1, "max": 32, "step": 1, "tooltip":"How many resamples for the end. -1 means constant."}),
                             "cfgpp" : ("BOOLEAN", {"default": True}),
                             }}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self,resample,resample_end,cfgpp):
        sampler = comfy.samplers.KSAMPLER(
            distance_wrap(resample=resample,cfgpp=cfgpp,resample_end=resample_end))
        return (sampler, )
