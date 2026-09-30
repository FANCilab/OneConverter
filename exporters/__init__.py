"""Discover and expose the concrete exporters in this package."""

from importlib import import_module
import inspect
import pkgutil

from .base import Exporter


def _discover_exporters() -> tuple[type[Exporter], ...]:
    """Import exporter modules and return their concrete exporter classes."""
    exporter_classes: set[type[Exporter]] = set()
    module_infos = sorted(pkgutil.iter_modules(__path__), key=lambda info: info.name)

    for module_info in module_infos:
        if module_info.ispkg or module_info.name == "base" or module_info.name.startswith("_"):
            continue

        module_name = f"{__name__}.{module_info.name}"
        try:
            module = import_module(module_name)
        except Exception as exc:
            raise ImportError(f"Failed to import exporter module {module_name!r}") from exc

        for _, exporter_class in inspect.getmembers(module, inspect.isclass):
            if (
                exporter_class is not Exporter
                and issubclass(exporter_class, Exporter)
                and not inspect.isabstract(exporter_class)
                and exporter_class.__module__ == module.__name__
            ):
                exporter_classes.add(exporter_class)

    return tuple(
        sorted(exporter_classes, key=lambda cls: (cls.__module__, cls.__qualname__))
    )


REGISTERED_EXPORTERS = _discover_exporters()

# Preserve direct imports such as ``from alf_toolbox.exporters import MpciExporter``.
globals().update({exporter.__name__: exporter for exporter in REGISTERED_EXPORTERS})

__all__ = [
    "Exporter",
    "REGISTERED_EXPORTERS",
    *(exporter.__name__ for exporter in REGISTERED_EXPORTERS),
]
