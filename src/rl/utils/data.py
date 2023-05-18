__all__ = [
    "unbatch_graph",
    "get_data_from_trainer",
    "get_dataset_from_loader",
    "get_adj",
    "graph_concat",
    "get_nb_samples_from_loader",
    "get_nb_samples_from_data",
    "sample_from_loader",
    "move_batch_to_device",
    "handle_device",
    "sample_members_and_nonmembers_from_loader",
    "extract_input_nodes",
    "create_nodeloader",
]

from loguru import logger  # pylint: disable=unused-import
import typing as ty

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, NodeLoader, DataLoader as GeoDataLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils.convert import from_networkx
import networkx as nx

from .err_msg import wrong_type_err


GRAPH_TYPE = ty.Union[nx.Graph, Data]
INPUT_TYPE = ty.Union[GRAPH_TYPE, ty.Sequence[GRAPH_TYPE]]
BATCH_TYPE = ty.Union[torch.Tensor, Data, ty.Sequence[torch.Tensor], ty.Sequence[Data]]


def handle_device(
    model: torch.nn.Module,
    batch: BATCH_TYPE,
) -> ty.Tuple[BATCH_TYPE, torch.device]:
    """Get device from model and move batch to it."""
    device: torch.device = next(model.parameters()).device
    assert isinstance(
        device, torch.device
    ), f"Property device was found of type {type(device)} but should be of type {torch.device}."
    batch = move_batch_to_device(batch, device)
    return batch, device


def move_batch_to_device(batch: BATCH_TYPE, device: torch.device) -> BATCH_TYPE:
    """Move batch to device."""
    try:
        if isinstance(batch, ty.Sequence):
            batch_new = []
            for _, x in enumerate(batch):
                if isinstance(x, (torch.Tensor, Data)):
                    batch_new.append(x.to(device))
                else:
                    batch_new.append(x)
            return batch_new
        return batch.to(device)
    except Exception as ex:
        logger.debug(f"Unable to move batch to device. {ex}")
        return batch


def get_data_from_trainer(
    trainer: pl.Trainer,
) -> ty.Sequence[ty.Optional[DataLoader]]:
    """Collects dataloaders from trainer."""
    # init
    train, val, test = None, None, None
    # training
    try:
        train = trainer.train_dataloader.loaders  # type: ignore
    except Exception:
        pass
    # validation
    try:
        val = trainer.val_dataloaders[0]  # type: ignore
    except Exception:
        pass
    # test
    try:
        test = trainer.test_dataloaders[0]  # type: ignore
    except Exception:
        pass
    return train, val, test


def extract_input_nodes(loader: ty.Union[NeighborLoader, NodeLoader]) -> torch.Tensor:
    """Extract input nodes from loader."""
    try:
        input_nodes = loader.input_data.args[0]
    except AttributeError:
        input_nodes = loader.input_data.node
    assert isinstance(input_nodes, torch.Tensor)
    return input_nodes


def sample_members_and_nonmembers_from_loader(
    loader: DataLoader,
    prior: float = 0.5,
    choose_nodes_based_on: ty.Dict[str, torch.Tensor] = None,
    **kwargs: ty.Any,
) -> ty.Tuple[DataLoader, DataLoader]:
    """Recreate dataloaders from the input one, containing random samples of it (non-overlapping).
    \n
    Args:
        loader (DataLoader):
            Input dataloader. Possible cases:\n
            * NeighborLoader | NodeLoader:\n
                This class has a `data` attribute storing the input graph and a `dataset` attribute which is just `range(0. N)` where `N` is the number of nodes. `loader.input_data.args[0]` will store a tensor of input node indeces that the loader will return. Thus, what this function will do is to recreate a new :class:`NeighborLoader` or :class:`NodeLoader` (this depends on your `torch_geometric` version), with its input nodes being `num_samples` samples taken from `loader.input_data.args[0]`.\n
            * DataLoader:\n
                For all other dataloaders, things should work in the same way: get `dataset` attribute from the loader, then `Subset(dataset, new_mask)`.\n
        prior (float):
            Membership prior.\n
        choose_nodes_based_on (ty.Dict[str, torch.Tensor], optional):\n
            Will choose member and non-member nodes only among nodes whose attributes defined in `choose_nodes_based_on.keys()` are equal to the values defined in `choose_nodes_based_on.values()`. For example: `choose_nodes_based_on = {'y': Tensor([0])}` will select only nodes whose node label `y` is `0`.\n
    """
    # NeighborLoader | NodeLoader
    if isinstance(loader, (NeighborLoader, NodeLoader)):
        # extract input nodes from loader
        input_nodes = extract_input_nodes(loader)
        assert isinstance(input_nodes, torch.Tensor)
        # check if we need to sample from a subset instead than from all nodes
        if choose_nodes_based_on is not None:
            data = loader.data
            mask = torch.ones(input_nodes.numel()).bool()
            for key, val in choose_nodes_based_on.items():
                attr = getattr(data, key, None)
                assert isinstance(attr, torch.Tensor)
                logger.trace(f"val: {val.size()}|{val.dim()}|{val.numel()}")
                if val.numel() == 0:
                    raise ValueError(f"Value is empty: {val}")
                if val.numel() == 1:
                    logger.trace(f"Reshaping {val}")
                    val = val.view(-1, 1).view(-1).repeat(attr.size(0))
                logger.trace(f"val: {val.size()}|{val.dim()}|{val.numel()}")
                tmp = attr == val
                mask = (mask * (tmp)).bool()
            input_nodes = input_nodes[mask]
        # uniformly sample n nodes from the loader
        n = input_nodes.numel()
        idx_in, idx_out = _sample_n_elements(n, prior)
        in_nodes = input_nodes[idx_in]
        out_nodes = input_nodes[idx_out]
        # create loaders
        kwargs.setdefault("batch_size", loader.batch_size)
        if isinstance(loader, NeighborLoader):
            # NeighborLoader
            kwargs.setdefault("num_neighbors", loader.node_sampler.num_neighbors)
            loader_in = NeighborLoader(loader.data, input_nodes=in_nodes.long(), **kwargs)  # type: ignore
            loader_out = NeighborLoader(loader.data, input_nodes=out_nodes.long(), **kwargs)  # type: ignore
            logger.debug(
                f"Created member loader with {in_nodes.numel()} samples and a non-member loader with {out_nodes.numel()} samples."
            )
        else:
            # NodeLoader
            loader_in = create_nodeloader(loader, in_nodes, **kwargs)
            loader_out = create_nodeloader(loader, out_nodes, **kwargs)
        return loader_in, loader_out
    # Generic DataLoader: must be the last if-statement
    if isinstance(loader, DataLoader):
        klass = type(loader)
        # get data
        dataset = loader.dataset
        # sample from it
        n: int = len(dataset)  # type: ignore
        idx_in, idx_out = _sample_n_elements(n, prior)
        dataset_in = Subset(dataset, idx_in.view(-1).numpy())
        dataset_out = Subset(dataset, idx_out.view(-1).numpy())
        # create loader
        kwargs.setdefault("batch_size", loader.batch_size)
        loader_in = klass(dataset_in, **kwargs)
        loader_out = klass(dataset_out, **kwargs)
        logger.debug(
            f"Created member loader with {len(dataset_in)} samples and a non-member loader with {len(dataset_out)} samples."
        )
        return loader_in, loader_out
    raise TypeError(f"Unsupported loader of type {type(loader)}")


def create_nodeloader(loader: NodeLoader, in_nodes: torch.Tensor, **kwargs: ty.Any) -> NodeLoader:
    """Recreate a NodeLoader from an existing one."""
    return NodeLoader(
        loader.data,
        node_sampler=NeighborSampler(loader.data, loader.node_sampler._num_neighbors.values),
        input_nodes=in_nodes.long(),
        **kwargs,
    )


def _sample_n_elements(
    n: int,
    prior: float = None,
    n_members: int = None,
) -> ty.Tuple[torch.Tensor, torch.Tensor]:
    """Sample N elements."""
    logger.debug(f"Found {n} samples.")
    n_members = int(n * prior) if n_members is None else n_members  # type: ignore
    logger.debug(f"Sampled {n_members} samples and {n-n_members} non-member samples.")
    idx = torch.randperm(n)
    idx_in = idx[:n_members]
    idx_out = idx[n_members:]
    return idx_in, idx_out


def sample_from_loader(
    loader: DataLoader,
    num_samples: int,
    replacement: bool = False,
    **kwargs: ty.Any,
) -> DataLoader:
    """Recreate a dataloader from the input one, containing `N` random samples of it.
    Args:
        loader (DataLoader):
            Input dataloader. Possible cases:
                * NeighborLoader | NodeLoader:
                    This class has a `data` attribute storing the input graph and a `dataset` attribute which is just `range(0. N)` where `N` is the number of nodes. `loader.input_data.args[0]` will store a tensor of input node indeces that the loader will return. Thus, what this function will do is to recreate a new :class:`NeighborLoader` or :class:`NodeLoader` (this depends on your `torch_geometric` version), with its input nodes being `num_samples` samples taken from `loader.input_data.args[0]`.
                * DataLoader:
                    For all other dataloaders, things should work in the same way: get `dataset` attribute from the loader, then `Subset(dataset, new_mask)`.
        num_samples (int):
            Number of samples to include.
        replacement (bool, optional):
            See :func:`torch.multinomial()`.
    """
    # NeighborLoader
    if isinstance(loader, (NeighborLoader, NodeLoader)):
        # get input graph
        data = loader.data
        # uniformly sample n nodes from the loader
        input_nodes = extract_input_nodes(loader)
        n_tot = input_nodes.numel()
        if num_samples > n_tot:
            num_samples = n_tot
        p = (0 * input_nodes + 1) / n_tot
        idx = p.multinomial(num_samples, replacement=replacement)
        assert idx.numel() == num_samples
        new_input_nodes = input_nodes[idx]
        assert new_input_nodes.numel() == num_samples
        # create loader
        kwargs.setdefault("batch_size", loader.batch_size)
        if isinstance(loader, NeighborLoader):
            kwargs.setdefault("num_neighbors", loader.node_sampler.num_neighbors)
            kwargs.setdefault("input_nodes", new_input_nodes.long())
            dataloader = NeighborLoader(data, **kwargs)  # type: ignore
        else:
            kwargs.setdefault("batch_size", loader.batch_size)
            dataloader = NodeLoader(
                data,
                node_sampler=NeighborSampler(
                    loader.data, loader.node_sampler._num_neighbors.values
                ),
                input_nodes=new_input_nodes.long(),
                **kwargs,
            )
        return dataloader  # type: ignore
    # Generic DataLoader: keep this one as last if-statement
    if isinstance(loader, DataLoader):
        klass = type(loader)
        # get data
        dataset = loader.dataset
        # sample from it
        n: int = len(dataset)  # type: ignore
        if num_samples < n:
            p = torch.Tensor(list(range(n)))
            mask = p.multinomial(num_samples, replacement=replacement)
            dataset = Subset(dataset, mask.view(-1).numpy())
        # create loader
        kwargs.setdefault("batch_size", loader.batch_size)
        return klass(dataset, **kwargs)
    raise TypeError(f"Unsupported loader of type {type(loader)}")


def graph_concat(graphs: INPUT_TYPE) -> GRAPH_TYPE:
    """Concat a list of graphs into one big graph."""
    # concat graphs into one big graph
    big_graph: GRAPH_TYPE
    if isinstance(graphs, (list, tuple)):
        graphs_: ty.List[Data] = [
            from_networkx(graph.to_undirected()) if isinstance(graph, nx.Graph) else graph
            for graph in graphs
        ]
        loader = GeoDataLoader(graphs_, batch_size=len(graphs))
        big_graph = next(iter(loader))
    else:
        big_graph = graphs
    wrong_type_err(big_graph, (nx.Graph, Data), extra=f"{big_graph}")
    return big_graph


def get_nb_samples_from_data(data: ty.Union[Dataset, DataLoader, torch.Tensor]) -> int:
    """Get number of samples."""
    if isinstance(data, torch.Tensor):
        return int(data.size(0))
    if isinstance(data, Dataset):
        return int(len(data))  # type: ignore
    return int(get_nb_samples_from_loader(data))


def get_nb_samples_from_loader(dataloader: DataLoader) -> int:
    """Extract dataset size from dataloder."""
    n_samples: int
    if isinstance(dataloader, NeighborLoader):
        n_samples = _get_samples_from_neighborloader(dataloader)
    else:
        n_samples = len(dataloader.dataset)  # type: ignore
    return int(n_samples)


def _get_samples_from_neighborloader(dataloader: NeighborLoader) -> int:
    """Get total number of samples."""
    if hasattr(dataloader, "input_nodes"):
        input_nodes: torch.Tensor = dataloader.input_nodes
        n_samples = _get_input_nodes(input_nodes)
    elif hasattr(dataloader, "data"):
        data: Data = dataloader.data
        input_nodes = data.train_mask
        n_samples = _get_input_nodes(input_nodes)
    else:
        raise AttributeError(
            f"Problem encountered. Cannot infer number of nodes in {dataloader.__class__.__name__}."
        )
    return n_samples


def _get_input_nodes(input_nodes: torch.Tensor) -> int:
    """Helper."""
    n_samples: int
    if isinstance(input_nodes, torch.LongTensor):
        n_samples = input_nodes.numel()
    elif isinstance(input_nodes, torch.BoolTensor):
        n_samples = int(input_nodes.sum())
    else:
        raise TypeError(f"Wrong type {type(input_nodes)} for input_nodes.")
    return n_samples


def unbatch_graph(
    graph: Data,
    num_neighbors: ty.List[int],
) -> ty.List[Data]:
    """Sampled nodes are sorted based on the order in which they were sampled. In particular, the first :obj:`batch_size` nodes represent the set of original mini-batch nodes."""
    wrong_type_err(graph, Data)
    out: ty.List[Data] = []
    for b in range(graph.batch_size):
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            batch_size=1,
            input_nodes=[b],
        )
        for s in loader:
            out.append(s)
    return out


def get_dataset_from_loader(dataset: ty.Union[Dataset, DataLoader]) -> Dataset:
    """Extracts a dataset object from a dataloader."""
    if isinstance(dataset, DataLoader):
        data_out = dataset.dataset
    elif isinstance(dataset, Dataset):
        data_out = dataset
    else:
        raise TypeError("Input dataset must be either a DataLoader or a Dataset.")
    return data_out


def get_adj(data: Data) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets adjacency matrix from edge index."""
    max_num_nodes = data.x.size(0)
    adj = np.zeros((max_num_nodes, max_num_nodes), dtype=np.int64)
    for idx in range(data.edge_index.shape[1]):
        i = data.edge_index[0, idx]
        j = data.edge_index[1, idx]
        adj[i, j] = 1
    adj = adj + adj.transpose()
    features = data.x
    labels = data.y
    return adj, features, labels
