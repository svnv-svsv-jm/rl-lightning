import sys
import time
from datetime import datetime
from tqdm import tqdm


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self,
            filename: str,
            mode: str = "a",
            time_info: bool = True,
        ):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
        self.time_info = time_info

    def write(self, message: str):
        # datetime object containing current date and time
        if self.time_info:
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            message = f"{now}  {message} "
        # write to outputs
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # If you want the output to be visible immediately

    def flush(self, *args, **kwargs):
        '''This flush method is needed for Python3 compatibility. This handles the flush command by doing nothing. You might want to specify some extra behavior here.
        '''
        self.log.flush() # If you want the output to be visible immediately



class Timer:
    def __init__(self,
            name: str,
            verbose: bool = True,
        ):
        self.name = name
        self.verbose = verbose

    def __enter__(self, *args, **kwargs):
        self.begin = time.time()
        if self.verbose: print(f'[{self.name}] In progress...')
        return self

    def __exit__(self, *args, **kwargs):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        if self.verbose: print(f'\tTime: {self.elapsed:7.3f}s or {time.strftime("%H:%M:%S", self.elapsedH)}')
