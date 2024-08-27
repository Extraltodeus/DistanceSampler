from .custom_samplers import distance_wrap

extra_samplers = {}
extra_samplers["heun_cfg_pp"] = distance_wrap(resample=1,cfgpp=True)

"""
To add a sampler to the list of samplers you can do it this way (outside of this comment):

extra_samplers["the_name_that_you_want"] = distance_wrap(resample=3,resample_end=-1,cfgpp=False)

"resample" is the starting value, how many more inferences it will use.
For resample_end "-1" means constant.
Constant resample at 0 gives Euler, 1 gives Heun.
cfgpp will determin if you want it or not. True or False.

You can remove the part below if you prefer to clean the list from the preset that I added.
"""

def make_preset(cfgpp,start,end):
    ppname = "_cfg_pp" if cfgpp else ""
    stepsn = "constant" if end == -1 else "fast"
    preset_name = f"distance_{stepsn}_{start}{ppname}"
    preset_sampler = distance_wrap(resample=start,resample_end=end,cfgpp=cfgpp)
    return preset_name, preset_sampler

ispp = [False,True] # CFGpp
resample_start = [3,4]
resample_const = [2,3,4]
for ipp in ispp:
    distance_p = resample_start
    for ep in distance_p:
        name, p_sampler = make_preset(ipp,ep,1)
        extra_samplers[name] = p_sampler
    distance_p = resample_const
    for ep in distance_p:
        name, p_sampler = make_preset(ipp,ep,-1)
        extra_samplers[name] = p_sampler