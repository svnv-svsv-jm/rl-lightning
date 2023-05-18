import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from rl.utils import class_for_name
from rl.buffer import Experience
from rl.datasets import RLDataset



class DM(pl.LightningDataModule):
    """Shared Datamodule class. Create and handle the dataloaders for training, validation and test. You can also shuffle the dataset here.
    """
    def __init__(self,
            dataclass: str = None,
            dataset: str = None,
            batch_size: int = 32,
            train_split: float = .8,
            shuffle: bool = True,
            seed: int = 0,
            # additional arguments to dataclass constructor
            dataclass_kwargs: dict = {},
            **kwargs,
        ):
        """
        Args:
            dataclass (str or data.Dataset):
                The name of the dataset class (data.Dataset) to instantiate.
            batch_size (int, optional):
                Defaults to 32.
            train_split (float, optional):
                Defaults to None. The percentage of training data. The rest goes for testing purposes.
            shuffle (bool, optional):
                Whether to shuffle the train/val split.
            seed (int, optional):
                Seed for shuffling.
        """
        super().__init__()
        # data
        assert (dataclass is not None) or (dataset is not None), f"Please provide either a `dataclass` or `dataset` keyword argument."
        self.dataclass = class_for_name(dataclass if dataset is None else dataset)
        # attr
        self.batch_size = batch_size
        self.train_split = train_split
        self.shuffle = shuffle
        self.seed = seed
        # datasets
        self.fit = None
        self.test = None
        # additional kwargs to be passed to the dataclass constructor
        self.kwargs = dataclass_kwargs

    def prepare_data(self, **kwargs):
        """Download data.
        """
        pass

    def setup(self, stage: str = None):
        # test data
        if (stage == 'test' or stage is None) and (self.test is None):
            print(f"[{self.__class__.__name__}] Setting up stage TEST...")
            self.test = self.dataclass(
                train=False,
                **self.kwargs,
            )
            if isinstance(self.test, RLDataset):
                self.shuffle = False
        # fit data
        if (stage == 'fit' or stage is None) and (self.fit is None):
            print(f"[{self.__class__.__name__}] Setting up stage FIT...")
            self.fit = self.dataclass(
                train=True,
                **self.kwargs,
            )
            # do this only if not iterableDataset
            if not isinstance(self.fit, RLDataset):
                # shuffle? shuffling here may be important: some dataclasses may have an indexing based on sorted labels, which means that you'll end up with only `self.train_split` percent of the labels in your training set, thus by shuffling we avoid this. this is different from the shuffling that takes place within each dataloader later on.
                if self.shuffle:
                    gen = torch.Generator().manual_seed(self.seed) # generator
                    idx = torch.randperm(len(self.fit), generator=gen)
                else:
                    idx = torch.cat(list(range(len(self.fit))))
                # split `self.fit` into training and validation data
                size_train = int(self.train_split * len(self.fit))
                size_val = len(self.fit) - size_train
                self.idx_train = idx[0:size_train].tolist()
                self.idx_val = idx[size_train:size_train + size_val].tolist()
            else:
                self.shuffle = False
                self.idx_train = None
                self.idx_val = None

    def train_dataloader(self, num_workers=0):
        dataset = Subset(self.fit, self.idx_train) if self.idx_train else self.fit
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, num_workers=0):
        dataset = Subset(self.fit, self.idx_val) if self.idx_val else self.fit
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def test_dataloader(self, num_workers=0):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self, *args, **kwargs):
        return self.test_dataloader(*args, **kwargs)

    def create_example_input_array(self, *args, **kwargs):
        if not isinstance(self.fit, RLDataset):
            example_input_array = iter(self.val_dataloader()).next()
        else:
            # create brand new dataset of class dataclass
            kwargs = self.kwargs.copy()
            kwargs['sample_size'] = 1
            dataset = self.dataclass(
                train=True,
                **kwargs,
            )
            # play
            state = dataset.env.reset()
            action = dataset.env.action_space.sample()
            new_state, reward, done, _ = dataset.env.step(action)
            exp = Experience(state, action, reward, done, new_state)
            # fill buffer with at least one tuple or you cannot iter(dataloader).next()
            dataset.buffer.append(exp)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                pin_memory=True,
            )
            # input example
            example_input_array = iter(dataloader).next()
        return example_input_array