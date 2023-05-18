import psutil
import GPUtil
import torch
import queue
import inspect
import sys
import time
import threading



def to_cuda(P, MEMLIM=0.80, is_global=True, verbose=False):
    """Safely puts a Tensor to the GPU. Checks if the GPU memory (of the process running the function or of the machine) ever goes beyond a given threshold.
    Args:
        P (Tensor): CPU Tensor to put to the GPU.
        MEMLIM (float, optional): memory threshold to respect (for the whole machine or just this process). Defaults to 0.80.
        is_global (boolean, optional): on a globabl memory basis or relative to this process. Defaults to True.
    Returns:
        P (Tensor): GPU Tensor.
    """
    verboseprint = print if verbose else lambda *a, **k: None
    func_name = inspect.currentframe().f_code.co_name
    if is_global:
        GPUs = GPUtil.getGPUs() # total
    else:
        TOTMEM = torch.cuda.get_device_properties(0).total_memory # of this process
    # create function and queue where to store returned argument
    que = queue.Queue()
    def fun(x, que):
        x = x.cuda()
        que.put(x)
    # check memory usage while thread is running, exit if too much memory is used
    t1 = threading.Thread(target=fun, args=(P,que))
    t1.start()
    verboseprint("[{func_name}] thread started")
    while t1.is_alive():
        if is_global:
            # on a global basis
            for GPU in GPUs:
                verboseprint(f"[{func_name}] Global GPU memory used {GPU.memoryUtil}")
                if GPU.memoryUtil > MEMLIM:
                    print(f"Global GPU memory used {GPU.memoryUtil} > {MEMLIM}, exiting...")
                    sys.exit(1)
        else:
            # process basis
            verboseprint(f"[{func_name}] GPU memory used by this process {mem/TOTMEM}")
            mem = torch.cuda.memory_allocated()
            if mem/TOTMEM > MEMLIM:
                print(f"GPU memory used by this process {mem/TOTMEM} > {MEMLIM}, exiting...")
                sys.exit(1)
        time.sleep(0.3)
    # you can calculate percentage of available memory
    TOT = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    verboseprint(f"[{func_name}] Global RAM memory used {TOT}")
    # recover return arg from queue
    P = que.get()
    return P


def show_gpu_usage(func_name=None, all=False):
    if func_name is None:
        func_name = "Info"
    mem = torch.cuda.memory_allocated()
    print(f"[{func_name}] GPU memory used by PyTorch: {round(mem/1e9, 2)}GB")
    GPUs = GPUtil.getGPUs()
    for GPU in GPUs:
        print(f"[{func_name}] Total memory used on GPU{GPU.id}: {round(GPU.memoryUtil*100)}%, {round(GPU.memoryUsed/1e9, 2)}GB")
    if all:
        GPUtil.showUtilization(all=True)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device