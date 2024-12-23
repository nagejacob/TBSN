import sys
sys.path.append('..')
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile
import time
import torch


if __name__ == '__main__':
    with get_accelerator().device(0):
        Model = getattr(__import__('network'), 'TBSN')
        model = Model().cuda()

        # t = time.time()
        # input = torch.zeros(1, 3, 256, 256).cuda()
        # for i in range(1000):
        #     with torch.no_grad():
        #         output = model(input)
        # print(time.time() - t), exit()


        flops, macs, params = get_model_profile(model=model,  # model
                                                input_shape=(1, 3, 256, 256),
                                                # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                args=None,  # list of positional arguments to the model.
                                                kwargs=None,  # dictionary of keyword arguments to the model.
                                                print_profile=False,
                                                # prints the model graph with the measured profile attached to each module
                                                detailed=False,  # print the detailed profile
                                                module_depth=-1,
                                                # depth into the nested modules, with -1 being the inner most modules
                                                top_modules=1,  # the number of top modules to print aggregated profile
                                                warm_up=0,
                                                # the number of warm-ups before measuring the time of each module
                                                as_string=False,
                                                # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                output_file=None,
                                                # path to the output file. If None, the profiler prints to stdout.
                                                ignore_modules=None)  # the list of modules to ignore in the profiling
        params = params / (1000000)
        flops = flops / (1000000000)
        print('params(M): ', params, 'flops(G): ', flops)
