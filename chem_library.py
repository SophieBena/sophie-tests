from dataclasses import dataclass
import numpy as np


class _SiteElementLibrary:
    # name, element_type, charge
    SITE_ELEMENT_DATA = [
        ['O', 'O', -2],
        ['Mg', 'Mg', +2],
        ['Fe2+', 'Fe', +2],
        ['Fe3+', 'Fe', +3],
        ['Ca', 'Ca', +2],
        ['K', 'K', +1],
        ['Al', 'Al', +3],
        ['Ti4+', 'Ti', +4],
        ['Ti3+', 'Ti', +3]
    ]

    def __init__(self):
        self.library = {}
        self._init_library()

    @classmethod
    def get_library(cls):
        return cls().library

    def _init_library(self):
        for elem_data in self.SITE_ELEMENT_DATA:
            self.add_site_element(*elem_data)


    def add_site_element(self, name, element_type, charge):
        self.library[name] = {'name': name, 'element_type': element_type,
                              'charge': charge}


SITE_ELEMENT_LIBRARY = _SiteElementLibrary.get_library()


@dataclass
class Atom:
    symbol: str
    name: str
    number: int
    weight: float = np.nan
    entropy: float = np.nan

    def __eq__(self, other):
        if type(other) is Atom:
            return self.symbol == other.symbol
        else:
            return self.symbol == other

@dataclass
class Element:
    H: Atom = Atom('H', 'hydrogen', 1, 1.0079, 130.68/2.0)
    He: Atom = Atom('He', 'helium', 2)

    #     , 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
    # 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    # 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
    # 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
    # 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
    # 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    # 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    # 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
    # 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    # 'Rf', 'Db', 'Sg'


# 4.00260, 6.94, 9.01218, 10.81, 12.011, 14.0067, 15.9994,
# 18.998403, 20.179, 22.98977, 24.305, 26.98154, 28.0855, 30.97376, 32.06,
# 35.453, 39.948, 39.102, 40.08, 44.9559, 47.90, 50.9415, 51.996, 54.9380,
# 55.847, 58.9332, 58.71, 63.546, 65.38, 69.735, 72.59, 74.9216, 78.96,
# 79.904, 83.80, 85.4678, 87.62, 88.9059, 91.22, 92.9064, 95.94, 98.9062,
# 101.07, 102.9055, 106.4, 107.868, 112.41, 114.82, 118.69, 121.75,
# 127.60, 126.9045, 131.30, 132.9054, 137.33, 138.9055, 140.12, 140.9077,
# 144.24, 145., 150.4, 151.96, 157.25, 158.9254, 162.50, 164.9304, 167.26,
# 168.9342, 173.04, 174.967, 178.49, 180.9479, 183.85, 186.207, 190.2,
# 192.22, 195.09, 196.9665, 200.59, 204.37, 207.2, 208.9804, 209., 210.,
# 222., 223., 226.0254, 227., 232.0381, 231.0359, 238.029, 237.0482, 244.,
# 243., 247., 247., 251., 254., 257., 258., 259., 260., 260., 260., 263.])

    # These entropy values are from Robie, Hemingway and Fisher (1979) USGS
    # Bull 1452 as stipulated by Berman (1988).  They are NOT the most recent
    # values (e.g.NIST)
    # DBL_MAX = 999999.0
    # PERIODIC_ENTROPES = ([
    #     130.68/2.0, 126.15, 29.12, 9.54, 5.90, 5.74, 191.61/2.0,
    #          205.15/2.0, 202.79/2.0, 146.32, 51.30, 32.68, 28.35, 18.81, 22.85,
    #     31.80, 223.08/2.0, 154.84, 64.68, 41.63, 34.64, 30.63, 28.91, 23.64,
    #     32.01, 27.28, 30.04, 29.87, 33.15, 41.63, 40.83, 31.09, 35.69, 42.27,
    #          245.46/2.0, 164.08, 76.78, 55.40, 44.43, 38.99, 36.40, 28.66, DBL_MAX,
    #     28.53, 31.54, 37.82, 42.55, 51.80, 57.84, 51.20, 45.52, 49.50,
    #          116.15/2.0, 169.68, 85.23, 62.42, 56.90, 69.46, 73.93, 71.09,
    #     DBL_MAX, 69.50, 80.79, 68.45, 73.30, 74.89, 75.02, 73.18, 74.01,
    #     59.83, 50.96, 43.56, 41.51, 32.64, 36.53, 32.64, 35.48, 41.63, 47.49,
    #     75.90, 64.18, 65.06, 56.74, DBL_MAX, DBL_MAX, 176.23, DBL_MAX, DBL_MAX,
    #     DBL_MAX, 53.39, DBL_MAX, 50.29, DBL_MAX, 51.46, DBL_MAX, DBL_MAX,
    #     DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX,
    #     DBL_MAX, DBL_MAX ])

