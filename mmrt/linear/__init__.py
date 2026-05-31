"""Linear model components for the MMRT pipeline.

The package init intentionally avoids eager submodule imports so lightweight
modules such as ``mmrt.linear.diagnostics`` and ``mmrt.linear.evaluate`` can be
imported without pulling in storage, model, or preprocessing dependencies.
"""

__all__: list[str] = []
