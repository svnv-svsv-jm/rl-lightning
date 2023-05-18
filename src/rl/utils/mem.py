import psutil



def mem_warning(threshold: float = 20, warn_only: bool = False):
    mem = psutil.virtual_memory()
    unit = 1024 * 1024 * 1024 # GB
    if mem.available <= threshold * unit:
        if warn_only:
            print(f"\n\n::WARNING:: Available memory is {mem.available/unit}GB \n\n")
        else:
            raise MemoryError(f"You cannot run this program because not enough memory is available ({mem.available/unit}GB) on this machine. This may cause server downtime.")
    else:
        print(f"\n\n::INFO:: Available memory is {mem.available/unit}GB \n\n")