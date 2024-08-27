from .custom_samplers import SamplerDistanceAdvanced
from .presets_to_add import extra_samplers

def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    added = 0
    samplers_names = [n for n in extra_samplers][::-1]
    for sampler in samplers_names:
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

add_samplers()

NODE_CLASS_MAPPINGS = {
    "SamplerDistance": SamplerDistanceAdvanced,
}