from abc import abstractmethod


# def composed(*decs):
#     def deco(f):
#         for dec in reversed(decs):
#             f = dec(f)
#         return f
#
#     return deco


def abstractproperty(fun):
    return classmethod(property(abstractmethod(fun)))
