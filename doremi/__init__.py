"""DoReMi-style domain reweighting over the patch-average clusters.

See `README.md` in this folder for the three-run pipeline. The modules are
importable on their own -- `domains` and `dro` need only numpy, so the domain
bookkeeping and the weight optimiser can be inspected without a GPU.
"""
