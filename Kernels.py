import math

"""
    Describes default kernel functions that can be used by keywords. 
    Use 'import kernels from Kernels' to iterate through the kernels.
"""

def uniform(val):
    if abs(val) < 1:
        return 0.5
    else:
        return 0

def triangular(val):
    if abs(val) < 1:
        return 1 - abs(val)
    else:
        return 0
 
def epanechnikov(val):
    if abs(val) < 1:
        t = 1 - val ** 2
        return 3 * t / 4
    else:
        return 0
 
def quatric(val):
    if abs(val) < 1:
        t = 1 - val ** 2
        return 15 * t * t / 16
    else:
        return 0
 
def triweight(val):
    if abs(val) < 1:
        t = 1 - val ** 2
        return 35 * t * t * t / 32
    else:
        return 0

def tricube(val):
    if abs(val) < 1:
        t = 1 - abs(val) * (val ** 2)
        return 70 * t * t * t / 81
    else:
        return 0

def gaussian(val):
    return 1 / math.sqrt(2 * math.pi) * math.exp(- val ** 2 / 2)
 
def cosine(val):
    if abs(val) < 1:
        return math.pi * ((10 ** 9 * math.cos(math.pi * val / 2)) / 10 ** 9) / 4
    else:
        return 0
 
def logistic(val):
    try:
        res = 1 / (math.exp(val) + 2 + math.exp(- val))
    except OverflowError:
        res = 0
    return res
 
def sigmoid(val):
    try:
        res = 2 / (math.pi * (math.exp(val) + math.exp(- val)))
    except OverflowError:
        res = 0
    return res

kernels = {
            'uniform': uniform,
            'triangular': triangular,
            'epanechnikov': epanechnikov,
            'quatric': quatric,
            'triweight': triweight,
            'tricube': tricube,
            'gaussian': gaussian,
            'cosine': cosine,
            'logistic': logistic,
            'sigmoid': sigmoid
            }
