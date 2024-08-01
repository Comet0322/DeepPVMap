import segmentation_models_pytorch as smp
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile

encoder_list = [
    "timm-efficientnet-b0",
    "timm-efficientnet-b5",
    "timm-efficientnet-b7",
    "mit_b0",
    "mit_b2",
    "mit_b5",
]

for encoder_name in encoder_list:
    with get_accelerator().device(3):
        model = smp.Unet(
            encoder_name,
            encoder_weights="imagenet",
            classes=1,
            activation="sigmoid",
        )
        batch_size = 1
        flops, macs, params = get_model_profile(
            model=model,
            input_shape=(
                batch_size,
                3,
                256,
                256,
            ),  # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
            args=None,  # list of positional arguments to the model.
            kwargs=None,  # dictionary of keyword arguments to the model.
            print_profile=False,  # prints the model graph with the measured profile attached to each module
            detailed=True,  # print the detailed profile
            module_depth=-1,  # depth into the nested modules, with -1 being the inner most modules
            top_modules=1,  # the number of top modules to print aggregated profile
            warm_up=10,  # the number of warm-ups before measuring the time of each module
            as_string=True,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
            output_file=None,  # path to the output file. If None, the profiler prints to stdout.
            ignore_modules=None,
        )  # the list of modules to ignore in the profiling

        print(encoder_name, macs, params)
