import torch
from comfy.k_diffusion.sampling import trange, to_d
import comfy.model_patcher
import comfy.samplers
from comfy.k_diffusion import sampling
from comfy import model_sampling
from math import pi
mmnorm = lambda x: (x - x.min()) / (x.max() - x.min())
selfnorm = lambda x: x / x.norm()
EPSILON = 1e-4

@torch.no_grad()
def matrix_batch_slerp(t, tn, w):
    dots = torch.mul(tn.unsqueeze(0), tn.unsqueeze(1)).sum(dim=[-1,-2], keepdim=True).clamp(min=-1.0 + EPSILON, max=1.0 - EPSILON)
    mask = ~torch.eye(tn.shape[0], dtype=torch.bool, device=tn.device)
    A, B, *rest = dots.shape
    rest_1s = (1,) * len(rest)
    dots = dots[mask].reshape(A, B - 1, *rest)
    omegas = dots.acos()
    sin_omega = omegas.sin()
    res = t.unsqueeze(1).repeat(1, B - 1, *rest_1s) * torch.sin(w.div(B - 1).unsqueeze(1).repeat(1, B - 1, *rest_1s) * omegas) / sin_omega
    res = res.sum(dim=[0, 1]).unsqueeze(0)
    return res

@torch.no_grad()
def fast_distance_weights(t, use_softmax=False, use_slerp=False, uncond=None):
    orig_shape = t.shape[1:]
    if t.shape[1] == 1 and t.ndim == 4:
        t = t.squeeze(1)
    elif t.ndim < 3:
        raise ValueError("Can't handle input with dimensions < 3")
    else:
        t = t.reshape(t.shape[0], -1, *t.shape[-2 if t.ndim > 3 else -1:])
        if t.ndim == 3:
            t = t.unsqueeze(-1)
        if uncond is not None:
            uncond = uncond.reshape(1, *t.shape[1:])
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
    return res if res.shape == orig_shape else res.reshape(orig_shape)

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

def get_ancestral_step_ext(sigma, sigma_next, eta=1.0, is_rf=False):
    if sigma_next == 0 or eta == 0:
        return sigma_next, sigma_next * 0.0, 1.0
    if not is_rf:
        return (*sampling.get_ancestral_step(sigma, sigma_next, eta=eta), 1.0)
    # Referenced from ComfyUI.
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_ip1, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
    sigma_up = (sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2)**0.5
    x_coeff = alpha_ip1 / alpha_down
    return sigma_down, sigma_up, x_coeff

def internal_step(x, d, dt, sigma, sigma_next, sigma_up, x_coeff, noise_sampler):
    x = x + d * dt
    if sigma_up == 0 or noise_sampler is None:
        return x
    noise = noise_sampler(sigma, sigma_next).mul_(sigma_up)
    if x_coeff != 1:
        # x gets scaled for flow models.
        x *= x_coeff
    return x.add_(noise)

def fix_step_range(steps, start, end):
    if start < 0:
        start = steps + start
    if end < 0:
        end = steps + end
    start = max(0, min(steps - 1, start))
    end = max(0, min(steps - 1, end))
    return (end, start) if start > end else (start, end)

# Euler and CFGpp part taken from comfy_extras/nodes_advanced_samplers
def distance_wrap(
    resample, resample_end=-1, cfgpp=False, sharpen=False, use_softmax=False,
    distance_first=0, distance_last=-1, eta_first=0, eta_last=-1, distance_eta_first=0, distance_eta_last=-1,
    use_slerp=False, perp_step=False, smooth=False, use_negative=False, eta=0.0, s_noise=1.0,
    distance_step_eta=0.0, distance_step_s_noise=1.0, distance_step_seed_offset=42,
):
    @torch.no_grad()
    def sample_distance_advanced(model, x, sigmas, eta=eta, s_noise=s_noise, noise_sampler=None, distance_step_noise_sampler=None, extra_args=None, callback=None, disable=None):
        nonlocal distance_first, distance_last, eta_first, eta_last, distance_eta_first, distance_eta_last

        extra_args = {} if extra_args is None else extra_args
        seed = extra_args.get("seed")
        dstep_noise_sampler = None if distance_step_eta == 0 else distance_step_noise_sampler or noise_sampler or sampling.default_noise_sampler(x, seed=seed + distance_step_seed_offset if seed is not None else None)
        noise_sampler = None if eta == 0 else noise_sampler or sampling.default_noise_sampler(x, seed=seed)
        is_rf = isinstance(model.inner_model.inner_model.model_sampling, model_sampling.CONST)
        uncond = None
        steps = len(sigmas) - 1

        distance_first, distance_last = fix_step_range(steps, distance_first, distance_last)
        eta_first, eta_last = fix_step_range(steps, eta_first, eta_last)
        distance_eta_first, distance_eta_last = fix_step_range(steps, distance_eta_first, distance_eta_last)

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
        for i in trange(steps, disable=disable):
            use_distance = distance_first <= i <= distance_last
            use_eta = eta_first <= i <= eta_last
            use_distance_eta = distance_eta_first <= i <= distance_eta_last
            sigma, sigma_next = sigmas[i:i + 2]
            sigma_down, sigma_up, x_coeff = get_ancestral_step_ext(sigma, sigma_next, eta=eta if use_eta else 0.0, is_rf=is_rf)
            sigma_up *= s_noise
            dstep_sigma_down, dstep_sigma_up, dstep_x_coeff = get_ancestral_step_ext(sigma, sigma_next, eta=distance_step_eta if use_distance_eta else 0.0, is_rf=is_rf)
            dstep_sigma_up *= distance_step_s_noise

            res_mul = progression(sigma)
            if resample_end >= 0:
                resample_steps = max(min(current_resample,resample_end),min(max(current_resample,resample_end),int(current_resample * res_mul + resample_end * (1 - res_mul))))
            else:
                resample_steps = current_resample

            denoised = model(x, sigma * s_in, **extra_args)
            total += 1

            if cfgpp and torch.any(uncond):
                d = to_d(x - denoised + uncond, sigma, denoised)
            else:
                d = to_d(x, sigma, denoised)

            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas, 'sigma_hat': sigma, 'denoised': denoised})
            dt = sigma_down - sigma
            dstep_dt = dstep_sigma_down - sigma

            if sigma_next == 0 or resample_steps == 0 or not use_distance:
                # Euler method
                x = internal_step(x, d, dt, sigma, sigma_next, sigma_up, x_coeff, noise_sampler)
                continue
            # not Euler method
            x_n = [d]
            for re_step in trange(resample_steps, initial=1, disable=disable or resample_steps < 2, leave=False, desc="    Distance"):
                x_new = internal_step(x, d, dstep_dt, sigma, sigma_next, dstep_sigma_up, dstep_x_coeff, dstep_noise_sampler)
                new_denoised = model(x_new, sigma_next * s_in, **extra_args)
                if smooth:
                    new_denoised = new_denoised.abs().pow(1 / new_denoised.std().sqrt()) * new_denoised.sign()
                    new_denoised = new_denoised.div(new_denoised.std().sqrt())
                total += 1
                if cfgpp and torch.any(uncond):
                    new_d = to_d(x_new - new_denoised + uncond, sigma_next, new_denoised)
                else:
                    new_d = to_d(x_new, sigma_next * s_in, new_denoised)
                x_n.append(new_d)
                if re_step == 0:
                    d = (new_d + d) / 2
                    continue
                u = uncond if (use_negative and uncond is not None and torch.any(uncond)) else None
                d = fast_distance_weights(torch.stack(x_n), use_softmax=use_softmax, use_slerp=use_slerp, uncond=u)
                if sharpen or perp_step:
                    if sharpen and d_prev is not None:
                        d = normalize_adjust(d, d_prev, 1)
                    elif perp_step and d_prev is not None:
                        d = diff_step(d, d_prev, 0.5)
                    d_prev = d.clone()
                x_n.append(d)
            x = internal_step(x, d, dt, sigma, sigma_next, sigma_up, x_coeff, noise_sampler)
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

class SamplerDistanceBase:
    _DISTANCE_OPTIONS = None # All options by default.
    _DISTANCE_PARAMS = {
        "resample": ("INT", {
            "default": 3, "min": -1, "max": 32, "step": 1,
            "tooltip": "0 all along gives Euler. 1 gives Heun.\nAnything starting from 2 will use the distance method.\n-1 will do remaining steps + 1 as the resample value. This can be pretty slow.",
        }),
        "resample_end": ("INT", {
            "default": -1, "min": -1, "max": 32, "step": 1,
            "tooltip": "How many resamples for the end. -1 means constant.",
        }),
        "cfgpp": ("BOOLEAN", {
            "default": True,
            "tooltip": "Controls whether to use CFG++ sampling. When enabled, you should set CFG to a fairly low value.",
        }),
        "eta": ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 32.0, "step": 0.01,
            "tooltip": "Controls the ancestralness of the main sampler steps. 0.0 means to use non-ancestral sampling. Note: May not work well with some of the other options.",
        }),
        "s_noise": ("FLOAT", {
            "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
            "tooltip": "Scale factor for ancestral noise added during sampling. Generally should be left at 1.0 and only has an effect when ancestral sampling is used.",
        }),
        "distance_step_eta": ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 32.0, "step": 0.01,
            "tooltip": "Experimental option that allows using ancestral sampling for the internal distance steps. When used, should generally be a fairly low value such as 0.25. 0.0 means to use non-ancestral sampling for the internal distance steps.",
        }),
        "distance_step_s_noise": ("FLOAT", {
            "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
            "tooltip": "Scale factor for ancestral noise added in the internal distance steps. Generally should be left at 1.0 and only has an effect when distance_step_eta is non-zero.",
        }),
        "use_softmax": ("BOOLEAN", {
            "default": False,
            "tooltip": "Rather than using a min/max normalization and an exponent will use a softmax instead.",
        }),
        "use_slerp": ("BOOLEAN", {
            "default": False,
            "tooltip": "Will SLERP the predictions instead of doing a weighted average. The difference is more obvious when using use_negative.",
        }),
        "perp_step": ("BOOLEAN", {
            "default": False,
            "tooltip": "Experimental, not yet recommended.",
        }),
        "use_negative": ("BOOLEAN", {
            "default": False,
            "tooltip": "Will use the negative prediction to prepare the distance scores. This tends to give images with less errors from my testing.",
        }),
        "smooth": ("BOOLEAN", {
            "default": False,
            "tooltip": "Not recommended, will make everything brighter. Not smoother.",
        }),
        "sharpen": ("BOOLEAN", {
            "default": False,
            "tooltip": "Not recommended, attempts to sharpen the results but instead tends to make things fuzzy.",
        }),
        "distance_first": ("INT", {
            "default": 0, "min": -10000, "max": 10000, "step": 1,
            "tooltip": "First step to use distance sampling. You can use negative values to count from the end. Note: Steps are zero-based.",
        }),
        "distance_last": ("INT", {
            "default": -1, "min": -10000, "max": 10000, "step": 1,
            "tooltip": "Last step to use distance sampling. You can use negative values to count from the end. Note: Steps are zero-based.",
        }),
        "eta_first": ("INT", {
            "default": 0, "min": -10000, "max": 10000, "step": 1,
            "tooltip": "First step to use ancestral sampling. Only applies when ETA is non-zero. You can use negative values to count from the end. Note: Steps are zero-based.",
        }),
        "eta_last": ("INT", {
            "default": -1, "min": -10000, "max": 10000, "step": 1,
            "tooltip": "Last step to use ancestral sampling. Only applies when ETA is non-zero. You can use negative values to count from the end. Note: Steps are zero-based.",
        }),
        "distance_eta_first": ("INT", {
            "default": 0, "min": -10000, "max": 10000, "step": 1,
            "tooltip": "First step to use ancestral sampling for the distance steps. Only applies when distance ETA is non-zero. You can use negative values to count from the end. Note: Steps are zero-based.",
        }),
        "distance_eta_last": ("INT", {
            "default": -1, "min": -10000, "max": 10000, "step": 1,
            "tooltip": "Last step to use ancestral sampling for the distance steps. Only applies when distance ETA is non-zero. You can use negative values to count from the end. Note: Steps are zero-based.",
        }),
    }

    @classmethod
    def INPUT_TYPES(s):
        if s._DISTANCE_OPTIONS is None:
            return {"required": s._DISTANCE_PARAMS.copy()}
        return {"required": {k: s._DISTANCE_PARAMS[k] for k in s._DISTANCE_OPTIONS}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, **kwargs):
        sampler = comfy.samplers.KSAMPLER(distance_wrap(**kwargs))
        return (sampler, )

class SamplerDistance(SamplerDistanceBase):
    _DISTANCE_OPTIONS = ("resample", "resample_end", "cfgpp")

class SamplerDistanceAdvanced(SamplerDistanceBase):
    pass # Includes all options by default.
