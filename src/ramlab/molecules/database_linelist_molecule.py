# TODO: Implement a superclass for CH4 and other molecules that are loaded from a HITRAN lin
from src.ramlab.molecules.hitran_linelist_molecule import LineListMolecule
from src.ramlab.util.decorators import abstractproperty


class DatabaseLineListMolecule(LineListMolecule):
    pass
