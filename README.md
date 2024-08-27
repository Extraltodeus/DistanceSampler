# DistanceSampler
Heuristic modification of the Heun sampler using normalized distances.

No readme yet but you can try already.

Work better with little amounts of steps. Starting at 4, recommanded 10. CFG at 1.0 with 4-8 steps can give interesting results.

Using AYS/AYS30/Simple/Karras for the schedulers.

Includes a "Heun CFG++" using the [alternative Euler CFGpp version found in ComfyUI](https://github.com/comfyanonymous/ComfyUI/blob/7df42b9a2364bae6822fbd9e9fa10cea2e319ba3/comfy_extras/nodes_advanced_samplers.py) slapped with the Heun part at the end.

This may or may not be a correct implementation but the results are too good to be ignored.

"fast_3" and "constant_2" are the best speed/quality deals.

the file "presets_to_add" can be easily edited to add/remove presets to the list.
