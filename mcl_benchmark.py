import numpy as np
import scipy.sparse as sp

from ..BinsparseFormat import BinsparseFormat

def _normalize(xp, matrix):
    col_sums = xp.sum(matrix, axis=0)
    col_sums = xp.maximum

    

    
    
