# **Explanation of a Custom Diffusion Model Sampling Method in ComfyUI**

(made using Gemini (and read by myself so you don't just hurt yourself reading potential garbage) and re-touched a little so it's readable)

I recommand to have the code on the side to follow.

But in short:

Similarily to Heun's method or some other samplers, this sampler uses the last prediction to re-compute a new one.

Here it loops and in each iteration uses the distances in between each values to create a new result.

The new result is made from a weighted average (or a slerp using the same weights) where the weights are related to the proportions of proximity.

---

The fast\_distance\_weights function calculates weights for a batch of tensors based on their pairwise distances in a normalized space.

It takes the input tensor t, boolean flags use\_softmax and use\_slerp, and an optional unconditional tensor uncond as arguments.

The function begins by calculating the matrix norm of the input tensor t:

    norm = torch.linalg.matrix_norm(t, keepdim=True).

The batch size is then obtained:

    n = t.shape[0]

The input tensor is normalized by dividing it by its norm:

    tn = t.div(norm)

This step projects the tensors onto a unit hypersphere. 

(Note: that's a super fancy way to say I divide by the norm/radius, juste to ensure a similar scale, also usable by the slerp after)

Next, the function calculates a distance metric between all pairs of normalized tensors:

    distances = (tn.unsqueeze(0) - tn.unsqueeze(1)).abs().sum(dim=0)

This involves creating all pairwise combinations, calculating the element-wise absolute difference, and then summing along the first dimension.

This results in a measure of dissimilarity between each pair of normalized tensors. This distance is then transformed:

    distances = distances.max(dim=0, keepdim=True).values - distances

This operation inverts the distances, so that smaller original distances result in larger values after the subtraction.  

If an unconditional tensor uncond is provided (which is relevant for Classifier-Free Guidance), it is also normalized:

    uncond = uncond.div(torch.linalg.matrix_norm(uncond, keepdim=True))

Then, the distance of each normalized tensor tn from the normalized unconditional tensor is calculated and added to the existing distances:

    distances += tn.sub(uncond).abs().div(n)
    
This step incorporates information about how far each conditional sample is from the unconditional sample into the weighting scheme.  

The function then proceeds to normalize the distances to obtain weights. If use\_softmax is True, the softmax function is applied:

    distances = distances.mul(n).softmax(dim=0)

Softmax converts the distances into a probability distribution. If use\_softmax is False, the distances are first normalized by dividing by their maximum value and then squared:

    distances = distances.div(distances.max(dim=0).values).pow(2).

These squared values are then further normalized by dividing by their sum: 

    distances = distances / distances.sum(dim=0).

This provides an alternative way to obtain weights that sum to 1\.  

Finally, the function combines the original tensors t using the calculated weights. If use\_slerp is True, the matrix\_batch\_slerp function is called:

    res = matrix_batch_slerp(t, tn, distances)

Otherwise, a simple weighted sum is performed, followed by a normalization and scaling step:

    res = (t * distances).sum(dim=0).unsqueeze(0); res = res.div(torch.linalg.matrix_norm(res, keepdim=True)).mul(norm.mul(distances).sum(dim=0).unsqueeze(0)).

This function provides a flexible way to compute weights based on the relationships between tensors and their proximity to an optional unconditional sample, and it offers a choice between spherical interpolation and a weighted sum for combining the tensors based on these weights. The inclusion of the uncond parameter strongly suggests a connection to Classifier-Free Guidance principles.  

---

The function _matrix\_batch\_slerp_ implements batched spherical linear interpolation (SLERP). SLERP is a technique used to interpolate between two points on a unit sphere along the great circle that connects them, maintaining a constant angular velocity. This is particularly useful for interpolating rotations or, more generally, directions in high-dimensional spaces.

The function is decorated with @torch.no\_grad(), indicating that the operations within it should not be tracked for gradient computation, as it is likely used during inference or sampling. The function takes three arguments: t, which likely represents the original batch of tensors; tn, which is likely the normalized version of t; and w, which probably contains the weights or interpolation factors for each tensor in the batch.  

The first step inside matrix\_batch\_slerp is the calculation of dot products between all pairs of normalized tensors in tn: 

    dots = torch.mul(tn.unsqueeze(0), tn.unsqueeze(1)).sum(dim=[-1,-2], keepdim=True).clamp(min=-1.0 + EPSILON, max=1.0 - EPSILON)

tn.unsqueeze(0) and tn.unsqueeze(1) add dimensions to create all pairwise combinations for element-wise multiplication.

The .sum(dim=\[-1,-2\], keepdim=True) operation then calculates the dot product between these pairs along their feature dimensions.

The .clamp() operation ensures that the resulting dot product values stay within the range of -1 to 1, which is necessary for the subsequent acos() function.

The dot product of two unit vectors is the cosine of the angle between them, so this step effectively calculates the cosine of the angles between all pairs of normalized tensors.  

Next, a mask is created to exclude the diagonal elements of the dot product matrix: 

    mask = ~torch.eye(tn.shape, dtype=torch.bool, device=tn.device)

torch.eye() creates an identity matrix, and the bitwise NOT operator \~ inverts it, resulting in a mask where the diagonal elements are False and all others are True. This mask is then used to select only the off-diagonal elements from the dots tensor: 

    dots = dots[mask].reshape(A, B - 1, C, D, E)

This step focuses the interpolation on the relationships between *different* tensors in the batch, not a tensor with itself.

The dimensions of the dots tensor are then unpacked and reshaped to prepare for the SLERP calculation.  

The angles between the normalized tensors are calculated using the arccosine function: omegas \= dots.acos(). The sine of these angles is then computed: 

    sin_omega = omegas.sin()

The core of the SLERP implementation follows:

    res = t.unsqueeze(1).repeat(1, B - 1, 1, 1, 1) * torch.sin(w.div(B - 1).unsqueeze(1).repeat(1, B - 1, 1, 1, 1) * omegas) / sin_omega

This formula calculates a weighted sum based on the angles and the provided weights w.

The original tensor t is prepared for batch interpolation by adding and repeating dimensions.

The weights w are also manipulated to match the dimensions and are divided by B \- 1, suggesting a normalization of the weights across the other tensors in the batch.

The final result is obtained by summing the interpolated values across the first two dimensions and then unsqueezing to restore the expected shape:

    res = res.sum(dim=).unsqueeze(0)

This function essentially performs a smooth, spherical interpolation between multiple matrices in a batch, using the provided weights to determine the contribution of each interpolated direction.

The exclusion of self-pairs in the dot product calculation indicates that the function is designed to aggregate or combine information from distinct entities within the batch.  

--- 

The distance\_wrap function acts as a decorator that takes several parameters to configure a custom sampling process.

These parameters include resample, resample\_end, cfgpp, sharpen, use\_softmax, first\_only, use\_slerp, perp\_step, smooth, and use\_negative.

These flags and integer values control various aspects of the sampling algorithm implemented in the inner function sample\_distance\_advanced.

The decorator returns this inner function, effectively creating a customized sampling function based on the provided configuration.  

The inner function sample\_distance\_advanced implements the core custom sampling logic.

It takes the diffusion model (model), the initial noisy latent tensor (x), a tensor of noise levels (sigmas), optional extra arguments (extra\_args), a callback function (callback), and a disable flag for the progress bar (disable). It begins by initializing extra\_args if it's None and sets the unconditional output variable uncond to None.  

If either the cfgpp or use\_negative flag is True, the function defines an inner function post\_cfg\_function. This function is designed to capture the unconditional denoised output from the model after it has been called. It takes a dictionary of arguments and stores the unconditional denoised output in the uncond variable before returning the conditional denoised output. This mechanism is then integrated into the model's options within the ComfyUI framework using comfy.model\_patcher.set\_model\_options\_post\_cfg\_function. This setup allows the sampler to access the unconditional output needed for Classifier-Free Guidance or related techniques.  

The function then finds the minimum and maximum positive sigma values and defines a progression function. This function likely controls how the number of resampling steps changes during the sampling process based on the current noise level. A variable d\_prev is initialized to None, which will likely store the denoised direction from the previous step for use in sharpening or perpendicular step operations. The number of resampling steps for each main sampling step is determined based on the resample and resample\_end parameters. A counter total is initialized to track the number of model evaluations, and a tensor of ones s\_in is created for scaling sigma values.  

The main sampling loop iterates through the sequence of sigma values. In each iteration, the current sigma value is obtained, and the number of resampling steps is determined. The diffusion model is called to denoise the current latent tensor. Based on the cfgpp flag and the availability of the unconditional output, the denoised direction d is calculated using either a modified CFG approach or the standard method via the to\_d function. If a callback function is provided, it is called with the current state. The step size to the next sigma value is calculated.  

Depending on whether the next sigma is zero, no resampling steps are to be performed, or if it's after the first step and first\_only is True, an Euler step is applied to update the latent tensor x. Otherwise, a resampling process is initiated. A list x\_n is created to store the denoised directions from the resampling steps. An inner loop runs for the specified number of resampling steps. In each inner step, an initial Euler step is taken, the model is called to denoise the new latent state, and an optional smoothing operation is applied if the smooth flag is True. The new denoised direction new\_d is calculated, and it is appended to the x\_n list. For the first resampling step, the denoised direction d is updated by averaging it with new\_d. For subsequent resampling steps, the fast\_distance\_weights function is called to combine the denoised directions in x\_n using the specified use\_softmax and use\_slerp flags and the optional unconditional output. If the sharpen or perp\_step flags are True and a previous denoised direction d\_prev exists, the corresponding (undefined in the snippet) functions normalize\_adjust or diff\_step are called. The current denoised direction d is then stored in d\_prev, and it is also appended to x\_n. After the resampling loop, the latent tensor x is updated using the combined denoised direction d and the step size dt. Finally, the function returns the fully denoised latent tensor.

| Parameter | Type | Description | Potential Impact on Sampling |
| :---- | :---- | :---- | :---- |
| resample | Integer | Number of resampling steps to perform at each main sampling step. \-1 for dynamic adjustment. | Controls the level of refinement at each step; higher values increase computation time but potentially quality. |
| resample\_end | Integer | End value for dynamic resampling step calculation. | Influences how the number of resampling steps changes over the course of the sampling process. |
| cfgpp | Boolean | Enables Classifier-Free Guidance++. | Improves adherence to the conditioning prompt, potentially enhancing output quality and coherence. |
| sharpen | Boolean | Enables a sharpening operation on the denoised direction. | Likely enhances the details and sharpness of the generated image. |
| use\_softmax | Boolean | Uses the softmax function for calculating distance weights. | Affects the distribution of weights assigned to different tensors based on their distances. |
| first\_only | Boolean | If true, only performs resampling at the first sampling step. | Could be used for specific optimization strategies or early-stage refinement. |
| use\_slerp | Boolean | Uses Spherical Linear Interpolation to combine weighted tensors in fast\_distance\_weights. | Provides a smooth interpolation between tensors, potentially leading to more coherent results. |
| perp\_step | Boolean | Enables a perpendicular step operation on the denoised direction. | Likely influences the direction of the denoising step, potentially avoiding artifacts or improving consistency. |
| smooth | Boolean | Enables a smoothing operation on the denoised output during resampling. | Likely reduces noise or artifacts in the intermediate denoised results. |
| use\_negative | Boolean | Incorporates the unconditional output into the distance weight calculation, potentially related to negative prompting strategies. | Influences the weighting based on the distance from the unconditional sample. |

In conclusion, the provided Python code defines a custom sampling method for diffusion models within the ComfyUI framework that incorporates several advanced techniques. The method utilizes spherical linear interpolation (matrix\_batch\_slerp) and a distance-based weighting scheme (fast\_distance\_weights) to refine the denoising process. The distance\_wrap decorator allows for extensive configuration of the sampling behavior through various parameters, and the inner sample\_distance\_advanced function orchestrates the iterative denoising, including dynamic resampling steps and the integration of Classifier-Free Guidance. This custom sampler likely aims to improve the quality and control over the generated samples by considering the geometric relationships between latent tensors and by iteratively refining the denoising direction using information from multiple model evaluations within each main sampling step. Further research and experimentation with the various parameters and the potentially integrated sharpening, perpendicular step, and smoothing operations would be necessary to fully understand the capabilities and benefits of this sophisticated sampling method.

#### **Sources des citations**

1. KSampler | ComfyUI Wiki, consulté le avril 27, 2025, [https://comfyui-wiki.com/en/comfyui-nodes/sampling/k-sampler](https://comfyui-wiki.com/en/comfyui-nodes/sampling/k-sampler)  
2. SamplerCustom \- ComfyUI Wiki, consulté le avril 27, 2025, [https://comfyui-wiki.com/en/comfyui-nodes/sampling/custom-sampling/sampler-custom](https://comfyui-wiki.com/en/comfyui-nodes/sampling/custom-sampling/sampler-custom)  
3. k-diffusion \- PyPI, consulté le avril 27, 2025, [https://pypi.org/project/k-diffusion/](https://pypi.org/project/k-diffusion/)
