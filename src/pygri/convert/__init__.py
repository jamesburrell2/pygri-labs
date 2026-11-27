"""Convert trained PyTorch models into spiking networks and NIR graphs."""

from pygri.convert.ann_to_snn import ConversionResult, convert_sequential

__all__ = ["ConversionResult", "convert_sequential"]
