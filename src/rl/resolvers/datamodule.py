__all__ = ["ResolveDatamodule", "resolve_datamodule"]


def resolve_datamodule(dataset_name: str) -> str:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""
    dataset_name = dataset_name.lower()
    if dataset_name in ["cora", "citeseer", "pubmed"]:
        return "planetoid.yaml"
    if dataset_name == "imdb":
        return "imdb.yaml"
    if dataset_name in ["cifar10"]:
        return "gnnbenchmark.yaml"
    if dataset_name in ["proteins_full", "proteins"]:
        return "tudataset.yaml"
    raise ValueError(f"Dataset {dataset_name} not installed yet.")


class ResolveDatamodule:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> str:
        return resolve_datamodule(self.name)
