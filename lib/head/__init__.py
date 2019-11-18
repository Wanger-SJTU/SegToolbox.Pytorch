
from .fcn import FCN8s
from .fcn import FCN16s
from .fcn import FCN32s

from .non_local_dot_product import NONLocalBlock1D as NonlocalDotProduct1D
from .non_local_dot_product import NONLocalBlock2D as NonlocalDotProduct2D
from .non_local_dot_product import NONLocalBlock3D as NonlocalDotProduct3D

from .non_local_gaussian import NONLocalBlock1D as NonlocalLocalGaussian1D
from .non_local_gaussian import NONLocalBlock2D as NonlocalLocalGaussian2D
from .non_local_gaussian import NONLocalBlock3D as NonlocalLocalGaussian3D

from .non_local_concatenation import NONLocalBlock1D as NonlocalLocalConcatenation1D
from .non_local_concatenation import NONLocalBlock2D as NonlocalLocalConcatenation2D
from .non_local_concatenation import NONLocalBlock3D as NonlocalLocalConcatenation3D

from .non_local_embedded_gaussian import NONLocalBlock1D as NonlocalEmbeddedGaussian1D
from .non_local_embedded_gaussian import NONLocalBlock2D as NonlocalEmbeddedGaussian2D
from .non_local_embedded_gaussian import NONLocalBlock3D as NonlocalEmbeddedGaussian3D

from .PSPNet import PSPHead
from .JPU import JPU

__all__ = ['heads']

def get_nonlocal_block(cfg):
    key = "Nonlocal{}{}D".format(cfg.NONLOCAL.TYPE, cfg.NONLOCAL.DIM)
    return nonlocal_opts[key.lower()](cfg)
    

heads = {
'none':None,
'jpu' :JPU,
'fcn8s':FCN8s,
'fcn16s':FCN16s,
'fcn32s':FCN32s,
'psphead':PSPHead,
'nonlocal': get_nonlocal_block
}

nonlocal_opts = {
'nonlocaldotproduct1d':NonlocalDotProduct1D,
'nonlocaldotproduct2d':NonlocalDotProduct2D,
'nonlocaldotproduct3d':NonlocalDotProduct3D,
'nonlocallocalgaussian1d':NonlocalLocalGaussian1D,
'nonlocallocalgaussian2d':NonlocalLocalGaussian2D,
'nonlocallocalgaussian3d':NonlocalLocalGaussian3D,
'nonlocallocalconcatenation1d':NonlocalLocalConcatenation1D,
'nonlocallocalconcatenation2d':NonlocalLocalConcatenation2D,
'nonlocallocalconcatenation3d':NonlocalLocalConcatenation3D,
'nonlocalembeddedgaussian1d':NonlocalEmbeddedGaussian1D,
'nonlocalembeddedgaussian2d':NonlocalEmbeddedGaussian2D,
'nonlocalembeddedgaussian3d':NonlocalEmbeddedGaussian3D
}
