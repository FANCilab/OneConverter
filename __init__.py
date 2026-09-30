"""Transform local experimental acquisitions into IBL ALF datasets."""

if __package__:
    from .core import Collection, IBLDataset, OneSessionConverter, explain_dataset
    from . import exporters as _exporters
else:
    # Pytest imports a repository-root __init__.py directly when the checkout
    # directory is not itself a valid Python identifier (for example, it has
    # hyphens). The editable package remains the canonical import path.
    from alf_toolbox.core import Collection, IBLDataset, OneSessionConverter, explain_dataset
    from alf_toolbox import exporters as _exporters


Exporter = _exporters.Exporter
REGISTERED_EXPORTERS = _exporters.REGISTERED_EXPORTERS

# Match the package-level API to the dynamically discovered concrete exporters.
globals().update({exporter.__name__: exporter for exporter in REGISTERED_EXPORTERS})

__all__ = [
    "Collection",
    "Exporter",
    "IBLDataset",
    "OneSessionConverter",
    "REGISTERED_EXPORTERS",
    "explain_dataset",
    *(exporter.__name__ for exporter in REGISTERED_EXPORTERS),
]
