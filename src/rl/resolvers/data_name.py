__all__ = ["get_data_name"]

import hydra
from torch.utils.data import Dataset


def get_data_name(dm_cfg: dict) -> str:
    """Creates a tag based on the kind of data. The input should look like:
    ```
    {
        '_target_': 'brainiac_2.datasets.LightningNodeDatamodule',
        'root': '${paths.data_dir}',
        'dataset': {
            '_target_': 'torch_geometric.datasets.Planetoid',
            'root': '${paths.data_dir}',
            'name': 'Cora',
            'split': 'full'
        },
        'batch_size': 4,
        'n_steps': 3,
        'n_sampled_nodes': 3000,
        'use_all': True,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True
    }
    ```
    """
    dataset_cfg: dict = dm_cfg["dataset"]
    dataset: Dataset = hydra.utils.instantiate(dataset_cfg)
    return f"{dataset}".strip().strip("()")
