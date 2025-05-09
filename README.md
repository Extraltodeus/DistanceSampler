# DistanceSampler

A custom experimental sampler based on relative distances. The first few steps are slower and then the sampler accelerates (the end is made with Heun).

The idea is to get a more precise start since this is when most of the work is being done.

[A more technical explaination](https://github.com/Extraltodeus/DistanceSampler/blob/main/sampler_explaination.md)

Pros:

 - Less body horror / merged fused people.
 - Little steps required (4-10, recommanded general use: 7 with beta or AYS)
 - Can sample simple subjects without unconditional prediction (meaning with a CFG scale at 1) with a good quality.

Cons:

 - Slow, which is also a plus. Relax, the image is generating ⛱ (but really since it requires little amounts of steps while giving a lesser percentage of horrors I personally prefer it)

Variations:

 - The variation having a "n" in the name stands for "negative" and makes use of the unconditional prediction so to determin the best output. The results may vary depending on your negative prompt. In general it seems to make less mistakes. This is what I sample with in general.

 - The "p" variation uses a comparison with each previous step so to enhance the result. In general things become smoother / less messy.

---

### Potential compatibility issue:

If any error was to relate to tensors shape, uncomment these two lines in the file "presets_to_add.py":

    extra_samplers["Distance_fast"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False)
    extra_samplers["Distance_fast_n"] = distance_wrap(resample=3,resample_end=1,cfgpp=False,sharpen=False,use_negative=True)

These are basically the same except they don't use a spherical interpolation at the end. The interpolation was made with latent spaces such as those used in Stable Diffusion in mind. These two alternatives use a weighted average instead (the difference is barely noticeable from my testing).

---

## Comparisons

Examples below are using the beta scheduler. **The amount of steps has been adjusted to match the duration** has this sampler is quite slow, yet requires little amounts of steps.

left: Distance, 7 steps

right: dpmpp2m, 20 steps

![combined_side_by_side](https://github.com/user-attachments/assets/65a66eba-d038-45fc-9648-79084cc1e011)



Distance, 10 steps:

![distance_10_steps](https://github.com/user-attachments/assets/32d7cf21-4c6e-45e1-892f-adc08a0cfa49)

Distance n, 10 steps:

![distance_n_10_steps](https://github.com/user-attachments/assets/8d41657a-7e21-4909-b03f-01afa532edf7)

DPM++SDE (gpu), 14 steps:

![dpmppsder_14steps](https://github.com/user-attachments/assets/8a7eab3d-8948-4df6-b51a-8f456ecc6980)


## Disabled Guidance (CFG=1)


CFG scale at 1 on a normal SDXL model (works for simple subjects):

![ComfyUI_00645_](https://github.com/user-attachments/assets/c9676d09-2c66-4d48-86b0-f0cc7c82569c)

![ComfyUI_00640_](https://github.com/user-attachments/assets/daf59ad3-4abf-4a0f-abdd-6e7cf423e6b7)

![ComfyUI_00632_](https://github.com/user-attachments/assets/515ad683-d841-4c95-b452-9263fdeb46f1)

Distance p with a CFG at 1 and 6 steps:

![ComfyUI_00695_](https://github.com/user-attachments/assets/4ff194ac-a0ad-4e10-9cd4-c8d6aa4e3d57)

![ComfyUI_00692_](https://github.com/user-attachments/assets/a5bfc880-b7a3-45b3-867d-82ca7560bf34)
