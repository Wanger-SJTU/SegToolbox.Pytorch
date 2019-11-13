
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

__all__ = ['heads']

heads = {
'fcn8s':FCN8s,
'fcn16s':FCN16s,
'fcn32s':FCN32s,
'PSPHead':PSPHead,
'NonlocalDotProduct1D':NonlocalDotProduct1D,
'NonlocalDotProduct2D':NonlocalDotProduct2D,
'NonlocalDotProduct3D':NonlocalDotProduct3D,
'NonlocalLocalGaussian1D':NonlocalLocalGaussian1D,
'NonlocalLocalGaussian2D':NonlocalLocalGaussian2D,
'NonlocalLocalGaussian3D':NonlocalLocalGaussian3D,
'NonlocalLocalConcatenation1D':NonlocalLocalConcatenation1D,
'NonlocalLocalConcatenation2D':NonlocalLocalConcatenation2D,
'NonlocalLocalConcatenation3D':NonlocalLocalConcatenation3D,
'NonlocalEmbeddedGaussian1D':NonlocalEmbeddedGaussian1D,
'NonlocalEmbeddedGaussian2D':NonlocalEmbeddedGaussian2D,
'NonlocalEmbeddedGaussian3D':NonlocalEmbeddedGaussian3D
}