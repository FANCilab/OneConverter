# AGENTS.md

## Project purpose

Build a Python tool that transforms experimental acquisitions into the IBL standard format.

## Project map

- `core.py`: primary class structure; `OneSessionConverter` is the main user-facing class. It produces IBL-standard formatted data; use the official `one` interface to discover and load the output rather than adding custom loading abstractions.
- `ONE/`: read-only checkout of the official ONE repository; import its `one` package directly.
- `IBL documentation/`: authoritative IBL standard documentation. The `.docx` defines terminology for docstrings. `Local Data Standards.pdf` describes the lab's preliminary NAS organization and is not part of the IBL standard.

## Coding conventions

- Follow the IBL documentation and existing `core.py`/official ONE patterns.
- Avoid local (nested) functions unless explicitly requested.
- Do not add validation or verification logic unless explicitly requested.
- Comment code thoroughly where practical.
- Keep `OneSessionConverter` simple and user-oriented.

## Adding an acquisition exporter

- Add one `.py` file under `exporters/` defining an `Exporter` subclass. Exporter discovery is automatic.
- Read `IBL documentation/` for dataset names, types, dimensions, units, and descriptions. Put each dataset in the class `datasets` map as an `IBLDataset` with explicit fields and a documentation string; use a clear placeholder when the documentation omits a description.
- Implement one module-level export function per dataset and `get_collections()` to expose only datasets supported by the session's source files. Use `Collection` for their ALF locations.
- Keep session paths and time zero in `OneSessionConverter`. Verify a real export with `one.search`, `one.list_datasets`, `one.list_collections`, `one.load_dataset`, `one.load_object`, `one.load_collection`, and `explain_dataset`.
