# DistanceSampler
Heuristic modification of the Heun sampler using a custom function based on normalized distances.

Work better with little amounts of steps. Starting at 4, recommanded 10. CFG at 1.0 with 4-8 steps can give interesting results.

Using AYS/AYS30/Simple/Karras for the schedulers.

Currently using the "fast" or the "s4e1" version with 10 steps [AYS30 scheduler](https://github.com/pamparamm/ComfyUI-ppm) and CFG scale at 8. It slays.


Also includes a "Heun CFG++" using the [alternative Euler CFGpp version found in ComfyUI](https://github.com/comfyanonymous/ComfyUI/blob/7df42b9a2364bae6822fbd9e9fa10cea2e319ba3/comfy_extras/nodes_advanced_samplers.py) slapped with the Heun part at the end. This may or may not be a correct implementation but the results are too good to be ignored.

the file "presets_to_add" can be easily edited to add/remove presets to the list.
