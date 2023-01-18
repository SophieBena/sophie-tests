from __future__ import annotations  # Enable Python 4 type hints in Python 3

import abc
from abc import ABC
import re
from collections import Counter
from typing import List, Set, Dict, Tuple, Optional

from dataclasses import dataclass, field

import numpy as np
from thermoengine.chem_library import SITE_ELEMENT_LIBRARY
from thermoengine import UnorderedList


__all__ = ['_CrystalSite', 'OxideMolComp']

class _OrderedCrystal:
    def __init__(self):
        self.sites = {}

    def add_site(self, site_name: str, potential_occupants: List[str],
                 multiplicity: float = 1, occupancy: str = None):

        site = _CrystalSite(potential_occupants, multiplicity=multiplicity)
        if occupancy is not None:
            site.occupancy = occupancy

        self.sites[site_name] = site

    def _get_site(self, site_name: str) -> _CrystalSite:
        if site_name in self.sites:
            return self.sites[site_name]
        else:
            raise self.MissingSiteError

    def _get_sites(self) -> List[_CrystalSite]:
        sites = [self._get_site(site) for site in self.sites]
        return sites

    def set_site_occupants(self, site_occupants: Dict[str, str]):
        for site_name in site_occupants:
            site = self._get_site(site_name)
            site.occupancy = site_occupants[site_name]

    def get_potential_site_occupants(self, site_name: str) -> List[str]:
        site = self._get_site(site_name)
        return site.potential_occupants

    def get_elemental_abundances(self) -> Dict[str, float]:
        sites = self._get_sites()

        if len(sites) == 0:
            raise self.EmptyCrystalError

        occupancy_info = self._get_site_occupancy_info(sites)
        elemental_abundances = self._sum_up_elemental_abundance(occupancy_info)
        return elemental_abundances

    @staticmethod
    def _sum_up_elemental_abundance(occupancy_info: Dict) -> Dict:
        abundances = dict.fromkeys(occupancy_info['potential'], 0)
        for elem, site_mult in zip(occupancy_info['current'],
                                   occupancy_info['multiplicity']):
            abundances[elem] = abundances[elem] + site_mult
        return abundances

    @staticmethod
    def _get_site_occupancy_info(sites: List[_CrystalSite]) -> Dict:
        occupancy_info = {'current': [], 'multiplicity': [], 'potential': []}
        for site in sites:
            potential_element_types = site.get_potential_element_types()
            current_element_type = site.current_occupant_element_type
            occupancy_info['current'].append(current_element_type)
            occupancy_info['multiplicity'].append(site.multiplicity)
            occupancy_info['potential'].extend(potential_element_types)

        return occupancy_info

    class MissingSiteError(Exception):
        pass

    class EmptyCrystalError(Exception):
        pass


class _CrystalSite:
    def __init__(self, potential_occupants: List[str],
                 multiplicity: float = 1):
        self._occupancy = _Occupancy(potential_occupants)
        self.multiplicity = multiplicity

    @property
    def potential_occupants(self) -> List[str]:
        return self._occupancy.potential_occupants

    @property
    def occupancy(self) -> str:
        return self._occupancy.current_occupant

    @occupancy.setter
    def occupancy(self, site_element: Union[str, dict]):
        self._occupancy.current_occupant = site_element

    @property
    def current_occupant_element_type(self) -> str:
        return self._occupancy.current_occupant_element_type

    def get_potential_element_types(self):
        return self.potential_occupants.element_types

    @property
    def composition(self):
        return self._occupancy.occupancy_values

class _Occupancy:
    def __init__(self, potential_occupants: List[str]):
        self._potential_occupants = _SiteElementGroup(potential_occupants)
        self.reinitialize_occupancy_values()
        self._occupancy_values[potential_occupants[0]] = 1.0

    @property
    def potential_occupants(self):
        return self._potential_occupants

    @property
    def occupancy_values(self):
        return self._occupancy_values

    @property
    def current_occupant(self):
        return self._get_fully_occupied_site_element()

    @current_occupant.setter
    def current_occupant(self, site_element_info: Union[str, dict]):

        if type(site_element_info) is dict:
            for site_elem in site_element_info.keys():
                if site_elem not in self.potential_occupants.names:
                    raise self.UnknownSiteOccupantError

            self._occupancy_values = site_element_info

        else:
            if site_element_info not in self.potential_occupants.names:
                raise self.UnknownSiteOccupantError

            self.reinitialize_occupancy_values()
            self._occupancy_values[site_element_info] = 1.0

    @property
    def current_occupant_element_type(self) -> str:
        element_types = self.potential_occupants.element_types
        current_occupant_index = self.potential_occupants.names.index(
            self._get_fully_occupied_site_element())
        return element_types[current_occupant_index]

    def _get_fully_occupied_site_element(self):
        """
        Clunky method for get fully occupied site element key - to be replaced

        """
        fully_occupied_key = [key for key, value in
                              self._occupancy_values.items()
                              if value == 1]
        return fully_occupied_key[0]

    def reinitialize_occupancy_values(self):
        self._occupancy_values = dict.fromkeys(
            self.potential_occupants.names, 0)

    class UnknownSiteOccupantError(Exception):
        pass

    def __eq__(self, other):
        # return sorted(list(self._elements)) == sorted(list(other))
        pass

class _SiteElementGroup:
    def __init__(self, site_element_names: List[str]):
        self._init_site_elements(site_element_names)

    def _init_site_elements(self, site_element_names: List[str]):
        self._elements = []
        for elem in site_element_names:
            self._elements.append(_SiteElement.get_element(elem))

    def get_index(self, index):
        return self.names[index]

    @property
    def names(self) -> List[str]:
        return [elem.name for elem in self._elements]

    @property
    def element_types(self) -> List[str]:
        return [elem.element_type for elem in self._elements]

    @property
    def charges(self) -> List[float]:
        return [elem.charge for elem in self._elements]

    def __eq__(self, other):
        # return sorted(list(self._elements)) == sorted(list(other))
        return sorted(self.names) == sorted(other)


class _SiteElement:
    SITE_ELEMENT_LIBRARY = SITE_ELEMENT_LIBRARY

    @classmethod
    def get_element(cls, element_symbol):
        if element_symbol not in cls.SITE_ELEMENT_LIBRARY.keys():
            raise cls.InvalidElementSymbol

        element_info = cls.SITE_ELEMENT_LIBRARY[element_symbol]
        element = cls(element_info['name'], element_info['charge'],
                      element_info['element_type'])
        return element

    @classmethod
    def get_group_of_elements(cls, element_symbols):
        elements = [cls.get_element(elem) for elem in element_symbols]
        return elements

    class InvalidElementSymbol(Exception):
        pass

    def __init__(self, name, charge, element_type):
        self.name = name
        self.charge = charge
        self.element_type = element_type

@dataclass(order=True)
class Comp(ABC):
    """Abstract base composition class"""
    sort_index : float =field(init=False, repr=False)
    elem_comp: ElemMolComp = field(init=False, repr=False)
    TOL = 1e-6
    # TOL: float = field(default=1e-6, repr=False, compare=False)


    @property
    @abc.abstractmethod
    def all_data(self) -> Dict[str, float]:
        pass



    @staticmethod
    def _remove_missing_components(comp):
        for component in list(comp.keys()):
            if comp[component] == 0:
                comp.pop(component)

        return comp

    @property
    def data(self) -> Dict[str, float]:
        return self._remove_missing_components(self.all_data)

    @property
    def data_is_empty(self) -> bool:
        return len(self.data) == 0

    @property
    def values(self) -> np.ndarray:
        return np.array(list(self.data.values()))

    @property
    def all_values(self) -> np.ndarray:
        return np.array(list(self.all_data.values()))

    @property
    def components(self) -> np.ndarray:
        return np.array(list(self.data.keys()))

    @property
    def zero_components(self) -> np.ndarray:
        return np.array(
            [this_component for this_component in self.all_components
             if this_component not in self.components])

    @property
    def all_components(self):
        return np.array(list(self.all_data.keys()))

    def normalize(self):
        amounts = np.array(list(self.all_data.values()))
        tot_amt = np.sum(amounts)
        comp_scaled = dict(zip(self.all_components, amounts/tot_amt))
        return self.__class__(**comp_scaled)

    def __add__(self, other):

        return self.__class__(**dict(Counter(self.data) +
                                Counter(other.data)))

    def __radd__(self, other):
        # filter empty Comp
        if other==0:
            other = ElemMolComp()

        return self.__class__(**dict(Counter(self.data) +
                                Counter(other.data)))

    def __mul__(self, other):
        scaled_comp = self.data
        for key, val in scaled_comp.items():
            scaled_comp[key] = other*val
        return self.__class__(**scaled_comp)

    def __rmul__(self, other):
        scaled_comp = self.data
        for key, val in scaled_comp.items():
            scaled_comp[key] = other*val
        return self.__class__(**scaled_comp)

    def _is_equals(self, other, this_class):
        other = self.create_comp_from_dict(other, this_class)

        if self.data_is_empty:
            return other.data_is_empty

        if type(other) is not this_class:
            return self.elem_comp.normalize()._approx_equals(other.elem_comp.normalize())


        return self.normalize()._approx_equals(other.normalize())

        # self_norm = self.normalize()
        # other_norm = other.nomralize()
        # return self_norm._approx_eq

    def create_comp_from_dict(self, other, this_class):
        if type(other) is dict:
            other = this_class(**other)
        return other

    def _approx_equals(self, other):
        if not sorted(self.components) == sorted(other.components):
            return False

        for key, val in self.all_data.items():
            if np.abs(other.all_data[key] - val) > self.TOL:
                return False

        return True



    def __eq__(self, other):
        return self._is_equals(other, self.__class__)

@dataclass(order=True)
class OxideMolComp(Comp):
    """
    Composition defined in terms of molar oxide amounts

    Works seamlessly with all other composition objects.
    """
    SiO2: float = 0.0
    TiO2: float = 0
    Al2O3: float = 0
    Fe2O3: float = 0
    Cr2O3: float = 0
    FeO: float = 0
    MnO: float = 0
    MgO: float = 0
    NiO: float = 0
    CoO: float = 0
    CaO: float = 0
    Na2O: float = 0
    K2O: float = 0
    P2O5: float = 0
    H2O: float = 0
    CO2: float = 0

    @property
    def all_data(self) -> Dict[str, float]:
        comp = self.__dict__.copy()
        comp.pop('elem_comp')
        comp.pop('sort_index')
        return comp

    def __eq__(self, other):
        return self._is_equals(other, self.__class__)

    def __post_init__(self):
        self.elem_comp = (
                self.SiO2 * Oxides.SiO2 +
                self.TiO2 * Oxides.TiO2 +
                self.Al2O3 * Oxides.Al2O3 +
                self.Fe2O3 * Oxides.Fe2O3 +
                self.Cr2O3 * Oxides.Cr2O3 +
                self.FeO * Oxides.FeO +
                self.MnO * Oxides.MnO +
                self.MgO * Oxides.MgO +
                self.NiO * Oxides.NiO +
                self.CoO * Oxides.CoO +
                self.CaO * Oxides.CaO +
                self.Na2O * Oxides.Na2O +
                self.K2O * Oxides.K2O +
                self.P2O5 * Oxides.P2O5 +
                self.H2O * Oxides.H2O +
                self.CO2 * Oxides.CO2
        )
        self.sort_index = self.elem_comp.sort_index

@dataclass(order=True)
class OxideWtComp(Comp):
    """
    Composition defined in terms of oxide weights

    Works seamlessly with all other composition objects.
    """
    SiO2: float = 0.0
    TiO2: float = 0
    Al2O3: float = 0
    Fe2O3: float = 0
    Cr2O3: float = 0
    FeO: float = 0
    MnO: float = 0
    MgO: float = 0
    NiO: float = 0
    CoO: float = 0
    CaO: float = 0
    Na2O: float = 0
    K2O: float = 0
    P2O5: float = 0
    H2O: float = 0
    CO2: float = 0

    @property
    def all_data(self) -> Dict[str, float]:
        comp = self.__dict__.copy()
        comp.pop('elem_comp')
        comp.pop('sort_index')
        return comp

    def __eq__(self, other):
        return self._is_equals(other, self.__class__)

    def __post_init__(self):
        self.elem_comp = (
                self.SiO2 / OxideWt.SiO2 * Oxides.SiO2 +
                self.TiO2 / OxideWt.TiO2 * Oxides.TiO2 +
                self.Al2O3 / OxideWt.Al2O3 * Oxides.Al2O3 +
                self.Fe2O3 / OxideWt.Fe2O3 * Oxides.Fe2O3 +
                self.Cr2O3 / OxideWt.Cr2O3 * Oxides.Cr2O3 +
                self.FeO / OxideWt.FeO * Oxides.FeO +
                self.MnO / OxideWt.MnO * Oxides.MnO +
                self.MgO / OxideWt.MgO * Oxides.MgO +
                self.NiO / OxideWt.NiO * Oxides.NiO +
                self.CoO / OxideWt.CoO * Oxides.CoO +
                self.CaO / OxideWt.CaO * Oxides.CaO +
                self.Na2O / OxideWt.Na2O * Oxides.Na2O +
                self.K2O / OxideWt.K2O * Oxides.K2O +
                self.P2O5 / OxideWt.P2O5 * Oxides.P2O5 +
                self.H2O / OxideWt.H2O * Oxides.H2O +
                self.CO2 / OxideWt.CO2 * Oxides.CO2
        )
        self.sort_index = self.elem_comp.sort_index

@dataclass(order=True)
class ElemMolComp(Comp):
    """
    Composition defined in terms of molar element amounts

    Provides root level support for intercomparing all composition
    objects of differing types. When in doubt, the molar quantities for
    each element are used.

    Works seamlessly with all other composition objects.
    """
    comp: Dict[str, float]
    elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
        'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
        'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
        'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
        'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
        'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
        'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg']

    def __init__(self, **kwargs):
        self.comp = kwargs
        self.elem_comp = self
        self._avg_atomic_num = self.calc_avg_atomic_number()
        self.sort_index = self._avg_atomic_num

    def calc_avg_atomic_number(self):
        val_tot = 0
        avg_atomic_num = 0
        for elem, val in self.data.items():
            val_tot += val
            atomic_num = self.elements.index(elem) + 1
            avg_atomic_num += atomic_num * val_tot
        if val_tot == 0:
            avg_atomic_num = 0
        else:
            avg_atomic_num /= val_tot
        return avg_atomic_num

    @property
    def all_data(self) -> Dict[str, float]:
        data = dict.fromkeys(self.elements, 0)
        data.update(self.comp)
        return data

    @classmethod
    def get_by_formula(cls, formula):
        elem_stoic = re.findall('[A-Z][^A-Z]*', formula)
        return ElemMolComp(**cls._extract_elem_count(elem_stoic))

    @staticmethod
    def _extract_elem_count(elem_stoic):
        comp = {}
        for ielem_stoic in elem_stoic:
            elem = re.findall('[A-z]+', ielem_stoic)[0]
            if ielem_stoic == elem:
                amt = 1
            else:
                amt = int(re.findall('[0-9]+', ielem_stoic)[0])
            comp[elem] = amt
        return comp

    def normalize(self):
        tot_amt = 1.0* np.sum(list(self.comp.values()))
        comp_scaled = self.comp.copy()
        for key, val in comp_scaled.items():
            comp_scaled[key] = val/tot_amt

        return ElemMolComp(**comp_scaled)

    def __repr__(self):
        class_name_str = f'{self.__class__.__name__}'
        sep = ', '
        sorted_key_val_pairs = [f'{key}: {float(val)}' for (key, val)
                                in sorted(self.comp.items())]
        all_key_val_str = sep.join(sorted_key_val_pairs)
        return class_name_str + '(' + all_key_val_str + ')'

    def __eq__(self, other):
        return self._is_equals(other, self.__class__)

@dataclass
class Oxides:
    """
    Convenient access to elemental composition of typical oxides

    Each oxide field works seamlessly with all other composition objects.
    """
    SiO2: ElemMolComp = ElemMolComp(Si=1, O=2)
    TiO2: ElemMolComp = ElemMolComp(Ti=1, O=2)
    Al2O3: ElemMolComp = ElemMolComp(Al=2, O=3)
    Fe2O3: ElemMolComp = ElemMolComp(Fe=2, O=3)
    Cr2O3: ElemMolComp = ElemMolComp(Cr=2, O=3)
    FeO: ElemMolComp = ElemMolComp(Fe=1, O=1)
    MnO: ElemMolComp = ElemMolComp(Mn=1, O=1)
    MgO: ElemMolComp = ElemMolComp(Mg=1, O=1)
    NiO: ElemMolComp = ElemMolComp(Ni=1, O=1)
    CoO: ElemMolComp = ElemMolComp(Co=1, O=1)
    CaO: ElemMolComp = ElemMolComp(Ca=1, O=1)
    Na2O: ElemMolComp = ElemMolComp(Na=2, O=1)
    K2O: ElemMolComp = ElemMolComp(K=2, O=1)
    P2O5: ElemMolComp = ElemMolComp(P=2, O=5)
    H2O: ElemMolComp = ElemMolComp(H=2, O=1)
    CO2: ElemMolComp = ElemMolComp(C=1, O=2)

@dataclass
class OxideWt:
    """
    Convenient access to molecular weights of typical oxides.
    """
    SiO2: float = 60.0848
    TiO2: float = 79.8988
    Al2O3: float = 101.96128
    Fe2O3: float = 159.6922
    Cr2O3: float = 151.9902
    FeO: float = 71.8464
    MnO: float = 70.9374
    MgO: float = 40.3044
    NiO: float = 74.7094
    CoO: float = 74.9326
    CaO: float = 56.0794
    Na2O: float = 61.97894
    K2O: float = 94.1954
    P2O5: float = 141.94452
    H2O: float = 18.0152
    CO2: float = 44.0098