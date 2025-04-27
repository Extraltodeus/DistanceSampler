import torch
from comfy.k_diffusion.sampling import trange, to_d
import comfy.model_patcher
import comfy.samplers
from math import pi
mmnorm = lambda x: (x - x.min()) / (x.max() - x.min())
selfnorm = lambda x: x / x.norm()
EPSILON = 1e-4

@torch.no_grad()
def matrix_batch_slerp(t, tn, w):
    dots = torch.mul(tn.unsqueeze(0), tn.unsqueeze(1)).sum(dim=[-1,-2], keepdim=True).clamp(min=-1.0 + EPSILON, max=1.0 - EPSILON)
    mask = ~torch.eye(tn.shape[0], dtype=torch.bool, device=tn.device)
    A, B, C, D, E = dots.shape
    dots = dots[mask].reshape(A, B - 1, C, D, E)
    omegas = dots.acos()
    sin_omega = omegas.sin()
    res = t.unsqueeze(1).repeat(1, B - 1, 1, 1, 1) * torch.sin(w.div(B - 1).unsqueeze(1).repeat(1, B - 1, 1, 1, 1) * omegas) / sin_omega
    res = res.sum(dim=[0, 1]).unsqueeze(0)
    return res

@torch.no_grad()
def fast_distance_weights(t, use_softmax=False, use_slerp=False, uncond=None):
    norm = torch.linalg.matrix_norm(t, keepdim=True)
    n  = t.shape[0]
    tn = t.div(norm)

    distances = (tn.unsqueeze(0) - tn.unsqueeze(1)).abs().sum(dim=0)
    distances = distances.max(dim=0, keepdim=True).values - distances

    if uncond != None:
        uncond = uncond.div(torch.linalg.matrix_norm(uncond, keepdim=True))
        distances += tn.sub(uncond).abs() #.div(n)

    if use_softmax:
        distances = distances.mul(n).softmax(dim=0)
    else:
        distances = distances.div(distances.max(dim=0).values).pow(2)
        distances = distances / distances.sum(dim=0)

    if use_slerp:
        res = matrix_batch_slerp(t, tn, distances)
    else:
        res = (t * distances).sum(dim=0).unsqueeze(0)
        res = res.div(torch.linalg.matrix_norm(res, keepdim=True)).mul(norm.mul(distances).sum(dim=0).unsqueeze(0))
    return res

@torch.no_grad()
def normalize_adjust(a,b,strength=1):
    c = a.clone()
    norm_a = a.norm(dim=1,keepdim=True)
    a = a / norm_a
    b = b / b.norm(dim=1,keepdim=True)
    d = mmnorm((a - b).abs())
    a = a - b * d * strength
    a = a * norm_a / a.norm(dim=1,keepdim=True)
    if a.isnan().any():
        a[~torch.isfinite(a)] = c[~torch.isfinite(a)]
    return a

# Euler and CFGpp part taken from comfy_extras/nodes_advanced_samplers
def distance_wrap(resample,resample_end=-1,cfgpp=False,sharpen=False,use_softmax=False,first_only=False,use_slerp=False,perp_step=False,smooth=False,use_negative=False):
    @torch.no_grad()
    def sample_distance_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        uncond = None

        if cfgpp or use_negative:
            uncond = None
            def post_cfg_function(args):
                nonlocal uncond
                uncond = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function)

        s_min, s_max = sigmas[sigmas > 0].min(), sigmas.max()
        progression = lambda x, y=0.5: max(0,min(1,((x - s_min) / (s_max - s_min)) ** y))
        d_prev = None

        if resample == -1:
            current_resample = min(10, sigmas.shape[0] // 2)
        else:
            current_resample = resample
        total = 0
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            sigma_hat = sigmas[i]

            res_mul = progression(sigma_hat)
            if resample_end >= 0:
                resample_steps = max(min(current_resample,resample_end),min(max(current_resample,resample_end),int(current_resample * res_mul + resample_end * (1 - res_mul))))
            else:
                resample_steps = current_resample

            denoised = model(x, sigma_hat * s_in, **extra_args)
            total += 1

            if cfgpp and torch.any(uncond):
                d = to_d(x - denoised + uncond, sigmas[i], denoised)
            else:
                d = to_d(x, sigma_hat, denoised)

            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat

            if sigmas[i + 1] == 0 or resample_steps == 0 or (i > 0 and first_only):
                # Euler method
                x = x + d * dt
            else:
                # not Euler method
                x_n = [d]
                for re_step in range(resample_steps):
                    x_new = x + d * dt
                    new_denoised = model(x_new, sigmas[i + 1] * s_in, **extra_args)
                    if smooth:
                        new_denoised = new_denoised.abs().pow(1 / new_denoised.std().sqrt()) * new_denoised.sign()
                        new_denoised = new_denoised.div(new_denoised.std().sqrt())
                    total += 1
                    if cfgpp and torch.any(uncond):
                        new_d = to_d(x_new - new_denoised + uncond, sigmas[i + 1], new_denoised)
                    else:
                        new_d = to_d(x_new, sigmas[i + 1] * s_in, new_denoised)
                    x_n.append(new_d)
                    if re_step == 0:
                        d = (new_d + d) / 2
                    else:
                        u = uncond if (use_negative and uncond is not None and torch.any(uncond)) else None
                        d = fast_distance_weights(torch.stack(x_n).squeeze(1), use_softmax=use_softmax, use_slerp=use_slerp, uncond=u)
                        if sharpen or perp_step:
                            if sharpen and d_prev is not None:
                                d = normalize_adjust(d, d_prev, 1)
                            elif perp_step and d_prev is not None:
                                d = diff_step(d, d_prev, 0.5)
                            d_prev = d.clone()
                        x_n.append(d)
                x = x + d * dt
        return x
    return sample_distance_advanced

def blend_add(t,v,s):
    tn = torch.linalg.norm(t)
    vn = torch.linalg.norm(v)
    vp = (v / vn - torch.dot(v / vn, t / tn) * t / tn) * tn
    return t + vp * s / 2

@torch.no_grad()
def diff_step(a, b, s):
    n = torch.linalg.matrix_norm(a, keepdim=True)
    x = a.div(n)
    y = b.div(torch.linalg.matrix_norm(b, keepdim=True))
    y = n * y.sub(x.mul(torch.mul(x, y).sum().clamp(min=-1.0, max=1.0)))
    return a - y * s

def perp_step_wrap(s=0.5):
    @torch.no_grad()
    def perp_step(model, x, sigmas, extra_args=None, callback=None, disable=None):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        previous_step = None
        for i in trange(len(sigmas) - 1, disable=disable):
            sigma_hat = sigmas[i]
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            if previous_step is not None and sigmas[i + 1] != 0:
                d = diff_step(d, previous_step, s)
            previous_step = d.clone()
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            x = x + d * dt
        return x
    return perp_step

# as a reference
@torch.no_grad()
def simplified_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

class SamplerDistanceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"resample": ("INT", {"default": 3, "min": -1, "max": 32, "step": 1,
                                                  "tooltip":"0 all along gives Euler. 1 gives Heun.\nAnything starting from 2 will use the distance method.\n-1 will do remaining steps + 1 as the resample value. This can be pretty slow."}),
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
