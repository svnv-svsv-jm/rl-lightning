__all__ = [
    "ResolveParams",
    "ResolveModel",
    "ResolveMetric",
    "resolve_metric",
    "resolve_model",
    "resolve_params",
]


def resolve_metric(name: str) -> str:
    """Resolve metric to optimize."""
    name = name.lower()
    if name in ["gcn"]:
        return "auroc/train"
    if name == ["vgae", "braignn"]:
        return "ap/train"
    raise ValueError(f"Model {name} not installed yet.")


def resolve_model(name: str) -> str:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""
    name = name.lower()
    if name in ["gcn"]:
        return "gcn.yaml"
    if name == "vgae":
        return "vgae.yaml"
    if name in ["braignn"]:
        return "braingnn.yaml"
    raise ValueError(f"Model {name} not installed yet.")


def resolve_params(name: str) -> str:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""
    name = name.lower()
    if name in ["gcn"]:
        return "gcn.yaml"
    if name == ["vgae", "braignn"]:
        return "lp.yaml"
    raise ValueError(f"Search space for model {name} not installed yet.")


class ResolveMetric:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> str:
        return resolve_metric(self.name)


class ResolveParams:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> str:
        return resolve_params(self.name)


class ResolveModel:
    """To be added as resolver: OmegaConf.register_new_resolver("fn", fn)"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> str:
        return resolve_model(self.name)
