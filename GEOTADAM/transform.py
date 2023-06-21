import abc, itertools
import numpy as np
from keras.preprocessing.image import apply_affine_transform

def get_transformer():
    # return Transformer(8, 8)
    return SimpleTransformer()
    
class Affine_Transformation(object):
    def __init__(self, flip, tx, ty, rotate):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.rotate = rotate
    def __call__(self, x):
        transformed_x = x.copy()
        if self.flip:
            transformed_x = np.fliplr(transformed_x)
        if self.tx != 0 or self.ty != 0:
            transformed_x = apply_affine_transform(transformed_x,
                                                   tx=self.tx, ty=self.ty,
                                                   row_axis=0, col_axis=1, channel_axis=2,
                                                   fill_mode="reflect")
        if self.rotate != 0:
            transformed_x = np.rot90(transformed_x, self.rotate)
        return transformed_x
             
class Abstract_Transformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()
    
    @property
    def n_transforms(self):
        return len(self._transformation_list)
    
    @abc.abstractmethod
    def _create_transformation_list(self):
        return
    
    def transform_batch(self, x_batch, indexes):
        assert len(x_batch) == len(indexes)
        
        transformed_batch = x_batch.copy()
        for i, index in enumerate(indexes):
            transformed_batch[i] = self._transformation_list[index](transformed_batch[i])
        return transformed_batch
        
class Transformer(Abstract_Transformer):
    def __init__(self, tx=8, ty=8):
        self.max_tx = tx
        self.max_ty = ty
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for flip, tx, ty, rotate in itertools.product((False, True),
                                                      (0, -self.max_tx, self.max_tx),
                                                      (0, -self.max_ty, self.max_ty),
                                                      range(4)):
            transformation = Affine_Transformation(flip, tx, ty, rotate)
            transformation_list.append(transformation)
        
        self._transformation_list = transformation_list
        return transformation_list
    
class SimpleTransformer(Abstract_Transformer):
    def _create_transformation_list(self):
        transformation_list = []
        for flip, rotate in itertools.product((False, True), range(4)):
            transformation = Affine_Transformation(flip, 0, 0, rotate)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list
        return transformation_list