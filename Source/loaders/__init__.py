"""File-format-specific loaders that produce numpy arrays for the
format-agnostic decode_sync_tags function.

Each loader in this package converts a recording from a particular
hardware/software format into the (n_channels, n_samples) numpy array
+ sample rate that decode_sync_tags expects, taking care of the
format's idiosyncrasies (header parsing, unit conversion to volts,
multi-file concatenation, etc.) so the decoder can stay
file-type-agnostic.
"""
