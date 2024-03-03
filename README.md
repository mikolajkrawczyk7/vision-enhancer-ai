# What is Vision Enhancer AI?
Vision Enhancer AI is a tool that improves the quality and fluidity of videos using advanced artificial intelligence models. It uses RIFE models to increase framerate by generating intermediate frames and RealESR models to increase resolution of videos. It runs on the ncnn framework, which uses the Vulkan API to enable fast and efficient processing on the GPU.

# Usage
> ./vision-enhancer-ai -i <input_path> -o <output_path> -m <model_path> <optional_model_path>
--verbose/-v               - Enable verbose mode.
--input/-i <input_path>    - Path to the input video/directory.
--output/-o <output_path>  - Path to the output video/directory.
--model/-m <model_path>    - Path to the model, must contain '<model_type>/<model_dir>/' at the end.
                             <model_type> = 'realesr'/'rife', <model_dir> = concrete model.
                             Example: 'models/rife/rife-v4.6/'.
--gpu/-g <gpu_id>          - GPU ID, '-1' for CPU or greater for GPU
--threads/-t <num_threads> - Number of processing threads.
--mul <output_mul>         - Target framerate multiplier. Example: '2' for 2x of input FPS.
--fps <target_fps>         - Target framerate, must be specified when input path is pointing to directory.
--scale/-s <output_scale>  - Target scale, this can be used to resize output video.
