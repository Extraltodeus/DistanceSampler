from .custom_samplers import distance_wrap

extra_samplers = {}
extra_samplers["heun_cfg_pp"] = distance_wrap(resample=1,cfgpp=True)

"""

To add a sampler to the list of samplers you can do it this way (outside of this comment):

extra_samplers["the_name_that_you_want"] = distance_wrap(resample=3,resample_end=-1,cfgpp=False)

"resample" is the starting value, how many more inferences it will use. More is slower.
For resample_end "-1" means constant.
Constant resample at 0 gives Euler, 1 gives Heun.
cfgpp will determin if you want it or not. True or False.

"""

extra_samplers["distance_fast"] = distance_wrap(resample=3,resample_end=1,cfgpp=False)
extra_samplers["distance_c2"] = distance_wrap(resample=2,resample_end=-1,cfgpp=False)
extra_samplers["distance_c3"] = distance_wrap(resample=3,resample_end=-1,cfgpp=False)
extra_samplers["distance_c4"] = distance_wrap(resample=4,resample_end=-1,cfgpp=False)
extra_samplers["distance_c2_cfg_pp"] = distance_wrap(resample=2,resample_end=-1,cfgpp=True)
extra_samplers["distance_fast_cfg_pp"] = distance_wrap(resample=3,resample_end=1,cfgpp=True)
