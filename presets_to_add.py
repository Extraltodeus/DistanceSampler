from .custom_samplers import distance_wrap, perp_step_wrap

extra_samplers = {}

"""

To add a sampler to the list of samplers you can do it this way (outside of this comment):

extra_samplers["the_name_that_you_want"] = distance_wrap(resample=3,resample_end=-1,cfgpp=False)

"resample" is the starting value, how many more inferences it will use. More is slower.
For resample_end "-1" means constant.
Constant resample at 0 gives Euler, 1 gives Heun.
Other variables:
 - use_negative: will use the negative prediction to prepare the distance scores. This tends to give images with less errors from my testing.
 - cfgpp: will determin if you want it or not. True or False.
 - use_slerp: will slerp the predictions instead of doing a weighted average. The difference is more obvious when using use_negative.
 - use_softmax: rather than using a min/max normalization and an exponent will use a softmax instead.
 - sharpen: not recommanded, attempts to sharpen the results but instead tends to make things fuzzy.
 - smooth: not recommanded, will make everything brighter. Not smoother.
 - perp step: experimental, not yet recommanded.
PerpStep is a test sampler, uncomment if you want to try.
"""
# Codename is Distance_fast_slerp
extra_samplers["Distance"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_slerp=True)
# Codename is Distance_fast_slerp_n
extra_samplers["Distance_n"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_slerp=True,use_negative=True)

# extra_samplers["PerpStep"] = perp_step_wrap(s=0.5)
# extra_samplers["euler_test"] = distance_wrap(resample=0,resample_end=-1)
# extra_samplers["heun_test"] = distance_wrap(resample=1,resample_end=1)
# extra_samplers["Distance_fast"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False)
# extra_samplers["Distance_fast_n"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_negative=True)
# extra_samplers["Distance_fast_slerp"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_slerp=True)
# extra_samplers["Distance_fast_slerp_n"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_slerp=True,use_negative=True)
# extra_samplers["Distance_fast_slerp_p"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_slerp=True,perp_step=True)
# extra_samplers["Distance_fast_slerp_np"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_slerp=True,use_negative=True,perp_step=True)

# # constant
# extra_samplers["distance_c2"] = distance_wrap(resample=2,resample_end=-1,cfgpp=False,sharpen=False)
# extra_samplers["distance_c3"] = distance_wrap(resample=3,resample_end=-1,cfgpp=False,sharpen=False)
# extra_samplers["distance_c4"] = distance_wrap(resample=4,resample_end=-1,cfgpp=False,sharpen=False)
# extra_samplers["Distance_cfgpp"] = distance_wrap(resample=4,resample_end=-1,sharpen=False,cfgpp=True)

# cfgpp
# extra_samplers["distance_step_cfg_pp"] = distance_wrap(resample=-1,resample_end=1,cfgpp=True)
extra_samplers["euler_cfg_pp_alt"] = distance_wrap(resample=0,cfgpp=True)
extra_samplers["heun_cfg_pp"] = distance_wrap(resample=1,cfgpp=True)
