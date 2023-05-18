from torch import nn



class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def _print(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def _warn(self, msg):
        print(f"[{self.__class__.__name__}:WARNING] {msg}")