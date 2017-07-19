
#matrix
import logging 
from logging import NullHandler 

log = logging.getLogger(__name__)
log.addHandler(NullHandler())

class Matrix(object):
    def __init__(self, obj):
        log.debug("Constructing matrix object")
        self._matrix = obj 
    
    def __getitem__(self, indices):
        return self._matrix([indices[0]][indices[1]])

    def __setitem__(self, indices, value):
        self._matrix([indices[0]][indices[1]] = value)

    @property 
    def shape(self):
        rows = len(self._matrix) 
        if rows == 0:
            rows = 1
            columns = 0
        else:
            columns = len(self._matrix[0])
        return (rows, columns) 

    def __abs__(self):
        result = Matrix([[abs(element) for element in row] for row in self._matrix)
        return result

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = [[element + other for element in row] for row in self._matrix]
        elif isinstance(other, Matrix):
            result = [[self[m, n] + other[m, n] for n in range(self.shape[1])]
                                                for m in range(self.shape[0])]
        else:
            raise TypeError
        return Matrix(result)

    def __mul_(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = [[element * other for element in row]
                                        for row in self._matrix]
        else:
            raise TypeError 
        return Matrix(result)

                    
        