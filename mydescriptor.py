
from nbnn import *

class Descriptor8Scale(Descriptor):
    """A descriptor which casts the values to a smaller range to save memory
    when you want to run a very large test.

    """
    def __call__(self, im_object):
        p, d = self.get_descriptors(im_object)
        # Convert to 8bit uints, take care of possible 512 values
        d=np.dstack([np.asarray(d)/2, \
            np.ones(d.shape,np.uint8)*127]).min(2).astype(np.uint8)
        return d

class Descriptor8Cut(Descriptor):
    """A descriptor which casts the values to a smaller range to save memory
    when you want to run a very large test.

    """
    def __call__(self, im_object):
        p, d = self.get_descriptors(im_object)
        # Convert to 8bit uints, cut high values (>255) to 255
        d[d>255]=255
        d.astype(np.uint8)
        return d
