# TODO: Implement a superclass for CH4 and other molecules that are loaded from a HITRAN lin
from ramlab.molecules.hitran_linelist_molecule import LineListMolecule
from ramlab.util.decorators import abstractproperty


class DatabaseLineListMolecule(LineListMolecule):
    pass
