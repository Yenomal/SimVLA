from __future__ import annotations
from typing import Dict, Type
from .base import DomainHandler
from .rmbench_hdf5 import RMBenchHDF5Handler

# Registry for dataset handlers
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    # RMBench
    "rmbench_hdf5": RMBenchHDF5Handler,
}


def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Add it to _REGISTRY in datasets/domain_handler/registry.py."
        )
