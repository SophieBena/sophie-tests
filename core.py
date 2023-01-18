"""The Core package implements general Python functions
required by the thermoengine package. Typically, these are focused on interfacing
with Objective-C vectors, arrays, matrices, etc.

"""
import collections

import numpy as np
from scipy import optimize
from scipy.optimize import minimize

# Objective-C imports
import ctypes
from ctypes import cdll
from ctypes import util
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from rubicon.objc import ObjCClass, NSObject, objc_method
if util.find_library('/usr/local/lib/libphaseobjc.dylib') is not None:
    cdll.LoadLibrary(util.find_library('/usr/local/lib/libphaseobjc.dylib'))
elif util.find_library('/usr/local/lib/libphaseobjc.so') is not None:
    cdll.LoadLibrary(util.find_library('/usr/local/lib/libphaseobjc.so'))

from collections import OrderedDict

# __all__ flag does not work in module file

__all__  = ['fill_array',
            'double_vector_to_array',
            'array_to_double_vector',
            'double_matrix_to_array',
            'make_scale_matrix',
            'get_src_object',
            'chem',
            'UnorderedList']

##################
# Array Handling #
##################
def fill_array(var1, var2):
    """Equilizes the dimension (shape) of two arrays or an array/scalar pair.

    This function is not normally called outside the equilibratepy module.

    On input ``var1`` and ``var2`` are either scalar/array pairs or arrays of the same shape.
    On output, the function returns a tuple with both arrays of the same shape.  A scalar is extended with
    contant entries if that action is required to match the dimension of a scalar/array pair. Two scalars
    are converted to two single dimension numpy arrays of length one.

    Uses the ``numpy.full_like`` function.

    Parameters
    ----------
    var1 : scalar or array
        If ``var1`` is an array, ``var2`` must be either a scalar or an array of the same shape.
    var2 : scalar or array
        If ``var2`` is an array, ``var1`` must be either a scalar or an array of the same shape.

    Returns
    -------
    result : tuple, (numpy array, numpy array)
        ``var1`` and ``var2`` converted to numpy arrays

    Examples
    --------

    >>> t = [500.0, 600.0]
    >>> p = 1000.0
    >>> t_a, p_a = fill_array(t, p)
    >>> print (t_a)
    [500.0, 600.0]
    >>> print (p_a)
    [1000.0, 1000.0]

    """
    var1_a = np.asarray( var1 )
    var2_a = np.asarray( var2 )

    if var1_a.shape==():
        var1_a = np.asarray( [var1] )
    if var2_a.shape==():
        var2_a = np.asarray( [var2] )

    # Begin try/except block to handle all cases for filling an array
    while True:
        try:
            assert var1_a.shape == var2_a.shape
            break
        except: pass
        try:
            var1_a = np.full_like( var2_a, var1_a )
            break
        except: pass
        try:
            var2_a = np.full_like( var1_a, var2_a )
            break
        except: pass

        # If none of the cases properly handle it, throw error
        assert False, 'var1 and var2 must both be equal shape or size=1'

    return var1_a, var2_a

################
#  Phase ObjC  #
################
def double_vector_to_array(vec):
    """Converts a DoubleVector Objective-C instance into a numpy 1-D array.

    This function is not normally called outside the equilibratepy module.

    Parameters
    ----------
    vec : an instance of the Objective-C class DoubleVector
        Contents of ``vec`` are a sequence of double precision entries.

    Returns
    -------
    array : numpy array
        Contents of ``vec`` as a 1-D numpy array

    """
    size = vec.size
    array = np.empty(size)
    m = vec.pointerToDouble()
    ctypes.cast(m, ctypes.POINTER(ctypes.c_double))
    for i in range(size):
        array[i] = m[i]
    return array

def array_to_double_vector(array):
    """Converts a 1-D numpy array into an instance of a DoubleVector Objective-C class.

    This function is not normally called outside the equilibratepy module.

    Parameters
    ----------
    array : an instance of a 1-D numpy array
        Contents of ``array`` must be a sequence of double precision entries.

    Returns
    -------
    vec : an instance of the Objective-C class DoubleVector
        Contents of ``array`` as a pointer to an instance of DoubleVector

    """
    doublevec_cls = ObjCClass('DoubleVector')
    # vec = (ctypes.c_double*array.size)()
    # ctypes.cast(vec, ctypes.POINTER(ctypes.c_double))
    vec = doublevec_cls.alloc().initWithSize_( array.size )
    vec_pointer = vec.pointerToDouble()
    for ind, val in enumerate(array):
        vec_pointer[ind] = val

    return vec

def array_to_ctype_array(np_array):
    """Converts a 1-D numpy array into a c-type array.

    Parameters
    ----------
    np_array : an instance of a 1-D numpy array
        Contents of ``array`` must be a sequence of double precision entries.

    Returns
    -------
    ctype_array : a c-type array

    """
    nc = len(np_array)
    m = (ctypes.c_double*nc)()
    ctypes.cast(m, ctypes.POINTER(ctypes.c_double))

    for i in range(np_array.size):
        m[i] = np_array[i]

    return m

def ctype_array_to_array(ctype_array):
    """Converts a c-type array into a numpy array

    Parameters
    ----------
    ctype_array : a c-type array
        Contents of ``array`` must be a sequence of double precision entries

    Returns
    -------
    np_array : an instance of a 1-D numpy array

    """
    N = ctype_array.size
    np_array = np.zeros(N)

    for i in range(N):
        np_array[i] = ctype_array.valueAtIndex_(i)

    return np_array

def double_matrix_to_array(mat):
    """Converts a DoubleMatrix Objective-C instance into a numpy 2-D array.

    This function is not normally called outside the equilibratepy module.

    Parameters
    ----------
    mat : an instance of the Objective-C class DoubleMatrix
        Contents of ``mat`` are a sequence of double precision entries organized as a matrix.

    Returns
    -------
    array : numpy array
        Contents of ``mat`` as a 2-D numpy array

    """
    Nrow, Ncol = mat.rowSize, mat.colSize
    array = np.empty((Nrow,Ncol))
    m = mat.pointerToPointerToDouble()
    ctypes.cast(m,ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    for i in range(Nrow):
        for j in range(Ncol):
            array[i,j] = m[i][j]
    return array

def double_tensor_to_array(ten):
    """Converts a DoubleTensor Objective-C instance into a numpy 3-D array.

    This function is not normally called outside the phases.py module.

    Parameters
    ----------
    ten : an instance of the Objective-C class DoubleTensor
        Contents of ``ten`` are a sequence of double precision entries organized as a 3x3 tensor

    Returns
    -------
    array : numpy array
        Contents of ``ten`` as a 3-D numpy array

    """
    N1st, N2nd, N3rd = ten.firstSize, ten.secondSize, ten.thirdSize
    array = np.empty((N1st,N2nd,N3rd))
    m = ten.pointerToPointerToPointerToDouble()
    ctypes.cast(m,ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))))
    for i in range(N1st):
        for j in range(N2nd):
            for k in range(N3rd):
                array[i,j,k] = m[i][j][k]
    return array

def get_src_object(classnm, return_class=False):
    """Initialize object from underlying source code.

    Parameters
    ----------
    classnm: str
        Name of src class
    return_class: bool, default False
        If True, return both src object and class as a tuple

    Returns
    -------
    src_obj: initialized src object
    src_cls: (if return_class is True) src class

    """
    src_cls = ObjCClass(classnm)
    src_obj = src_cls.alloc().init()

    if return_class:
        return src_obj, src_cls
    else:
        return src_obj

########
# Math #
########
def make_scale_matrix(array):
    scl_mat_a = np.dot(np.expand_dims(array,-1),
                       np.expand_dims(array,0))
    return scl_mat_a

#############
# Chemistry #
#############

class _Chem:
    OXIDE_ORDER = np.array([
        'SiO2','TiO2','Al2O3','Fe2O3','Cr2O3','FeO','MnO','MgO','NiO',
        'CoO','CaO','Na2O','K2O','P2O5','H2O','CO2'])
    PERIODIC_ORDER = np.array([
        None, 'H', 'He', 'Li', 'Be',  'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
        'Mg', 'Al', 'Si', 'P', 'S',   'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
        'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
        'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
        'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
        'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
        'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg' ])
    PERIODIC_NAMES = np.array([
        None, 'hydrogen', 'helium', 'lithium',  'beryllium', 'boron',
        'carbon', 'nitrogen', 'oxygen', 'fluorine', 'neon', 'sodium',
        'magnesium', 'aluminum', 'silicon', 'phosphorous', 'sulfur',
        'chlorine', 'argon', 'potassium', 'calcium', 'scandium', 'titanium',
        'vanadium', 'chromium', 'manganese', 'iron', 'cobalt', 'nickel',
        'copper', 'zinc', 'gallium', 'germanium', 'arsenic', 'selenium',
        'bromine', 'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium',
        'niobium', 'molybdenum', 'technetium', 'ruthenium', 'rhodium',
        'palladium', 'silver', 'cadmium', 'indium', 'tin', 'antimony',
        'tellurium', 'iodine', 'xenon', 'cesium', 'barium', 'lantahnum',
        'cerium', 'praseodymium', 'neodymium', 'promethium', 'samarium',
        'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium',
        'thulium', 'ytterbium', 'lutetium', 'hafnium', 'tantalum', 'tungsten',
        'rhenium', 'osmium', 'iridium', 'platinum', 'gold', 'mercury',
        'thallium', 'lead', 'bismuth', 'polonium', 'astatine', 'radon',
        'francium', 'radium', 'actinium', 'thorium', 'protactinium', 'uranium',
        'neptunium', 'plutonium', 'americium', 'curium', 'berkelium',
        'californium', 'einsteinium', 'fermium', 'mendelevium', 'nobelium',
        'lawrencium', 'ruferfordium', 'dubnium', 'seaborgium' ])
    PERIODIC_WEIGHTS = np.array([
        0.0, 1.0079, 4.00260, 6.94, 9.01218, 10.81, 12.011, 14.0067, 15.9994,
        18.998403, 20.179, 22.98977, 24.305, 26.98154, 28.0855, 30.97376, 32.06,
        35.453, 39.948, 39.102, 40.08, 44.9559, 47.90, 50.9415, 51.996, 54.9380,
        55.847, 58.9332, 58.71, 63.546, 65.38, 69.735, 72.59, 74.9216, 78.96,
        79.904, 83.80, 85.4678, 87.62, 88.9059, 91.22, 92.9064, 95.94, 98.9062,
        101.07, 102.9055, 106.4, 107.868, 112.41, 114.82, 118.69, 121.75,
        127.60, 126.9045, 131.30, 132.9054, 137.33, 138.9055, 140.12, 140.9077,
        144.24, 145., 150.4, 151.96, 157.25, 158.9254, 162.50, 164.9304, 167.26,
        168.9342, 173.04, 174.967, 178.49, 180.9479, 183.85, 186.207, 190.2,
        192.22, 195.09, 196.9665, 200.59, 204.37, 207.2, 208.9804, 209., 210.,
        222., 223., 226.0254, 227., 232.0381, 231.0359, 238.029, 237.0482, 244.,
        243., 247., 247., 251., 254., 257., 258., 259., 260., 260., 260., 263.])
    # These entropy values are from Robie, Hemingway and Fisher (1979) USGS
    # Bull 1452 as stipulated by Berman (1988).  They are NOT the most recent
    # values (e.g.NIST)
    DBL_MAX = 999999.0
    PERIODIC_ENTROPES = ([
        0.0, 130.68/2.0, 126.15, 29.12, 9.54, 5.90, 5.74, 191.61/2.0,
        205.15/2.0, 202.79/2.0, 146.32, 51.30, 32.68, 28.35, 18.81, 22.85,
        31.80, 223.08/2.0, 154.84, 64.68, 41.63, 34.64, 30.63, 28.91, 23.64,
        32.01, 27.28, 30.04, 29.87, 33.15, 41.63, 40.83, 31.09, 35.69, 42.27,
        245.46/2.0, 164.08, 76.78, 55.40, 44.43, 38.99, 36.40, 28.66, DBL_MAX,
        28.53, 31.54, 37.82, 42.55, 51.80, 57.84, 51.20, 45.52, 49.50,
        116.15/2.0, 169.68, 85.23, 62.42, 56.90, 69.46, 73.93, 71.09,
        DBL_MAX, 69.50, 80.79, 68.45, 73.30, 74.89, 75.02, 73.18, 74.01,
        59.83, 50.96, 43.56, 41.51, 32.64, 36.53, 32.64, 35.48, 41.63, 47.49,
        75.90, 64.18, 65.06, 56.74, DBL_MAX, DBL_MAX, 176.23, DBL_MAX, DBL_MAX,
        DBL_MAX, 53.39, DBL_MAX, 50.29, DBL_MAX, 51.46, DBL_MAX, DBL_MAX,
        DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX,
        DBL_MAX, DBL_MAX ])

    """
    Mole oxide to element conversion matrix
        - rows = oxides in standard OXIDE_ORDER
        - columns = elements in same order as they appear in OXIDE_ORDER matrix;
        thus, conversion matrix only valid for elements present in OXIDE_ORDER
        matrix; all Fe is converted to total Fe in column 4
        - column order = Si, Ti, Al, Fe, Cr, Mn, Mg, Ni, Co, Ca, Na, K, P, H, C, O

    """
    MOL_OXIDE_TO_ELEM = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                                    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 5],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]])

    LEPR_phase_symbols = {
        'Liquid':'Liq',
        'Clinopyroxene':'Cpx',
        'Garnet':'Grt',
        'Olivine':'Ol',
        'Orthopyroxene':'Opx',
        'Biotite':'Bt',
        'Fluid':None,
        'Corundum':'Crn',
        'Rutile':'Rt',
        'Plagioclase':'Fsp',
        'Amphibole':'Cam',
        'Zoisite':'Zo',
        'Cordierite':'Crd',
        'Muscovite':'Ms',
        'Quartz':'Qz',
        'Kyanite':'Ky',
        'Potassium feldspar':'Fsp',
        'Sillimanite':'Sil',
        'Spinel':'SplS',
        'Staurolite':None ,
        'Melilite':'Mll',
        'Carbonate melt':None,
        'Nepheline':'NphS',
        'Ilmenite':'Ilm',
        'Eskolaite':None,
        'Anorthite':'An',
        'cc-dol':None}

    def __init__(self):
        self._init_oxide_props()
        pass

    def _init_oxide_props(self):
        """Dictionary of oxide properties

        Returns
        -------
        oxide_data: dict with keys
            `oxides` : list of oxide strings
            `cations` : list of cation strings
            `molwt` : array of molecular weights
            `cat_num` : array of cation numbers
            `oxy_num` : array of oxygen numbers
            `oxycat_ratio` : array of oxygen/cation ratios

        """
        def make_oxide_dat(name, cation, molwt, charge, catnum, oxynum):
            oxycat_ratio = oxynum/catnum
            oxide_dat = {'name':name, 'cation':cation, 'molwt':molwt,
                         'charge':charge, 'catnum':catnum, 'oxynum':oxynum,
                         'oxycat_ratio': oxycat_ratio}
            return oxide_dat

        oxide_data = []
        oxide_data.append(make_oxide_dat('SiO2', 'Si', 60.0848, +4, 1, 2))
        oxide_data.append(make_oxide_dat('TiO2', 'Ti', 79.8988, +4, 1, 2))
        oxide_data.append(make_oxide_dat('Al2O3', 'Al', 101.96128, +3, 2, 3))
        oxide_data.append(make_oxide_dat('Fe2O3', 'Fe', 159.6922, +3, 2, 3))
        oxide_data.append(make_oxide_dat('Cr2O3', 'Cr', 151.9902, +3, 2, 3))
        oxide_data.append(make_oxide_dat('FeO', 'Fe', 71.8464, +2, 1, 1))
        oxide_data.append(make_oxide_dat('MnO', 'Mn', 70.9374, +2, 1, 1))
        oxide_data.append(make_oxide_dat('MgO', 'Mg', 40.3044, +2, 1, 1))
        oxide_data.append(make_oxide_dat('NiO', 'Ni', 74.7094, +2, 1, 1))
        oxide_data.append(make_oxide_dat('CoO', 'Co', 74.9326, +2, 1, 1))
        oxide_data.append(make_oxide_dat('CaO', 'Ca', 56.0794, +2, 1, 1))
        oxide_data.append(make_oxide_dat('Na2O', 'Na', 61.97894, +1, 2, 1))
        oxide_data.append(make_oxide_dat('K2O', 'K', 94.1954, +1, 2, 1))
        oxide_data.append(make_oxide_dat('P2O5', 'P', 141.94452, +5, 2, 5))
        oxide_data.append(make_oxide_dat('H2O', 'H', 18.0152, +1, 2, 1))
        oxide_data.append(make_oxide_dat('CO2', 'C', 44.0098, +4, 1, 2))


        oxide_props = OrderedDict()
        oxide_props['oxide_num'] = len(oxide_data)
        oxide_props['oxides'] = np.array([idat['name'] for idat in oxide_data])
        oxide_props['cations'] = np.array([idat['cation'] for idat in oxide_data])
        oxide_props['molwt'] = np.array([idat['molwt'] for idat in oxide_data])
        oxide_props['charge'] = np.array([idat['charge'] for idat in oxide_data])
        oxide_props['cat_num'] = np.array([idat['catnum'] for idat in oxide_data])
        oxide_props['oxy_num'] = np.array([idat['oxynum'] for idat in oxide_data])
        oxide_props['oxycat_ratio'] = np.array([idat['oxycat_ratio']
                                               for idat in oxide_data])

        # for idat in oxide_data:
        #     oxide = idat['name']
        #     oxide_props[oxide] = idat

        self._oxide_props = oxide_props
        pass

    @property
    def oxide_props(self):
        return self._oxide_props

    def select_oxides(self, oxide_names, oxide_values):
        # oxide_molwt = chem.oxide_props['molwt']
        oxides = chem.oxide_props['oxides']
        assert np.all([oxname in oxides for oxname in oxide_names]),(
            'oxide_names must all be valid oxide names')
        value = np.squeeze(np.array([oxide_values[oxides==iname]
                           for iname in oxide_names]))
        return value

    def calc_mol_oxide_comp(self, element_comp):
        major_cations = self.oxide_props['cations']

        # NOTE: not necessarily actually monovalent, but where we consider only one valence state
        monovalent_oxide_ind = np.where(self.oxide_props['cations']!='Fe')[0]
        FeO_oxide_ind = np.where(self.oxide_props['oxides']=='FeO')[0]
        Fe2O3_oxide_ind = np.where(self.oxide_props['oxides']=='Fe2O3')[0]

        def get_atomic_indices(monovalent_cations):
            monovalent_elem_ind = np.array([np.where(elems==icat)[0][0]
                                           for icat in monovalent_cations])
            oxy_elem_ind = np.where(elems=='O')[0][0]
            Fe_elem_ind = np.where(elems=='Fe')[0][0]

            return monovalent_elem_ind, oxy_elem_ind, Fe_elem_ind

        def calc_Fe_oxides(Fe_remain, oxy_remain):
            if Fe_remain == 0:
                mol_FeO = 0
                mol_Fe2O3 = 0

            else:
                #ratio = oxy_remain/Fe_remain
                #frac_Fe2O3 = (ratio-1)/(1.5-1)
                #frac_FeO = 1-frac_Fe2O3

                #mol_Fe_oxide = Fe_remain/(2*frac_Fe2O3 + 1*frac_FeO)
                #mol_FeO = mol_Fe_oxide*frac_FeO
                #mol_Fe2O3 = mol_Fe_oxide*frac_Fe2O3

                mol_Fe2O3 = oxy_remain - Fe_remain
                mol_FeO = 3.0*Fe_remain - 2.0*oxy_remain

            return mol_FeO, mol_Fe2O3

        monovalent_oxides = self.oxide_props['oxides'][monovalent_oxide_ind]
        monovalent_cations = self.oxide_props['cations'][monovalent_oxide_ind]
        monovalent_cat_num = self.oxide_props['cat_num'][monovalent_oxide_ind]
        monovalent_oxy_num = self.oxide_props['oxy_num'][monovalent_oxide_ind]

        elems = self.PERIODIC_ORDER

        monovalent_elem_ind, oxy_elem_ind, Fe_elem_ind =     get_atomic_indices(monovalent_cations)


        # extract array of all major elements from the composition
        monovalent_element_comp = element_comp[monovalent_elem_ind]
        monovalent_mol_oxide_comp = monovalent_element_comp/monovalent_cat_num

        monovalent_mol_oxy_tot = np.sum(monovalent_oxy_num*monovalent_mol_oxide_comp)


        oxy_remain = element_comp[oxy_elem_ind] - monovalent_mol_oxy_tot
        Fe_remain = element_comp[Fe_elem_ind]

        mol_FeO, mol_Fe2O3 = calc_Fe_oxides(Fe_remain, oxy_remain)

        mol_oxide_comp = np.zeros(self.oxide_props['oxide_num'])
        mol_oxide_comp[monovalent_oxide_ind] = monovalent_mol_oxide_comp
        mol_oxide_comp[FeO_oxide_ind] = mol_FeO
        mol_oxide_comp[Fe2O3_oxide_ind] = mol_Fe2O3

        return mol_oxide_comp

    def format_mol_oxide_comp(self, mol_oxides, convert_grams_to_moles=False):
        """
        convert mol_oxide dictionary to mol_oxide array

        Parameters:
        ==========
        mol_oxides: Dictionary
            Dictionary of molar oxide compositions with oxide
            names as keys

        convert_grams_to_moles: False, default
            boolean flag indicating weight input in grams

        Returns:
        ========
        mol_oxide_comp: np array
            Molar oxide array in standard oxide order

        """
        OXIDE_ORDER = self.OXIDE_ORDER
        mol_oxide_comp = np.zeros(len(OXIDE_ORDER))

        assert np.all([oxide in OXIDE_ORDER
                       for oxide in mol_oxides])

        for oxide in mol_oxides:
            mol_oxide = mol_oxides[oxide]
            ind, = np.where(OXIDE_ORDER==oxide)
            if convert_grams_to_moles:
                mol_oxide_comp[ind] = mol_oxide/self.oxide_props['molwt'][ind]
            else:
                mol_oxide_comp[ind] = mol_oxide

        return mol_oxide_comp

    def mol_oxide_to_elem(self, mol_oxides, oxide_names=None):
        """
        Convert mole oxide composition to mole element composition.

        Parameters
        ----------
        mol_oxides : array
            mole oxide composition defined in standard OXIDE_ORDER

        Returns
        -------
        mol_elem : array
            mole element composition with elements in same order as oxide array;
            only functions for elements present in OXIDE_ORDER; Fe given as total Fe
            elem order is Si, Ti, Al, Fe, Cr, Mn, Mg, Ni, Co, Ca, Na, K, P, H, C, O

        """
        # MOL_OXIDE_TO_ELEM = self.MOL_OXIDE_TO_ELEM


        MOL_OXIDE_TO_ELEM, oxide_names = self._validate_oxides_list(
            self.MOL_OXIDE_TO_ELEM, oxide_names)

        # oxide_molecular_wts = self.oxide_props['molwt']
        # oxide_molecular_wts = self.get_molwt(oxide_names)
        # MOL_OXIDE_TO_ELEM = self.select_oxides(oxide_names, self.MOL_OXIDE_TO_ELEM)


        mol_elem = np.dot(mol_oxides, MOL_OXIDE_TO_ELEM)

        return mol_elem

    def get_Berman_formula(self, element_comp):
        """Get chemical formula in Berman form (e.g., H(2.0)O(1.0)).

        Parameters
        ----------
        element_comp : double array
            Element composition defined in standard PERIODIC_ORDER

        Returns
        -------
        formula : str

        """
        formula = ''
        for amt, sym in zip(element_comp, self.PERIODIC_ORDER):
            if amt > 0.0:
                formula += sym + '(' + str(amt) + ')'

        return formula

    def elem_to_oxide(self):
        # %el x (oxide molecular weight/el weight) = wt% oxide

        raise NotImplemented

    def oxide_to_elem(self, oxide_names, oxide_wts):
        #oxide_names = np.array([oxide_names])
        #oxide_wts = np.array([oxide_wts])
        #oxide_molecular_wts = self.oxide_props['molwt']
        #wt% oxide to el% is -- wt% oxide x (el weight/oxide molecular weight)=el%

        raise NotImplemented

    def get_comp_subset(self):
        raise NotImplemented

    def _validate_oxides_list(self, oxide_values, oxide_names):
        oxide_values = np.array(oxide_values)
        ndim = oxide_values.ndim
        if ndim==2:
            noxides = oxide_values.shape[1]
        else:
            noxides = len(oxide_values)

        if oxide_names is None:
            oxide_names = self.OXIDE_ORDER

        oxide_names = np.array(oxide_names)

        assert len(oxide_names)==noxides, (
            'Num. of oxide names must match oxide wts.'
        )
        return oxide_values, oxide_names

    def _normalize_oxide_comp(self, oxide_values):
        if oxide_values.ndim == 2:
            totals = np.sum(oxide_values, axis=1)[:,np.newaxis]
        else:
            totals = np.sum(oxide_values)

        oxide_values = oxide_values/totals

        return oxide_values

    def wt_to_mol_oxide(self, oxide_wts, oxide_names=None):
        oxide_wts, oxide_names = self._validate_oxides_list(
            oxide_wts, oxide_names)

        # oxide_molecular_wts = self.oxide_props['molwt']
        # oxide_molecular_wts = self.get_molwt(oxide_names)
        oxide_molecular_wts = self.select_oxides(oxide_names, self.oxide_props['molwt'])

        mol_oxides = oxide_wts/oxide_molecular_wts
        mol_oxides = self._normalize_oxide_comp(mol_oxides)
        return mol_oxides

    def mol_to_wt_oxide(self, mol_oxides, oxide_names=None):
        mol_oxides, oxide_names = self._validate_oxides_list(
            mol_oxides, oxide_names)

        oxide_molecular_wts = self.oxide_props['molwt']

        wt_oxides = mol_oxides*oxide_molecular_wts
        wt_oxides = self._normalize_oxide_comp(wt_oxides)
        return wt_oxides

    def get_phase_symbols(self, rxn_data):
        return rxn_data['phase_symbols']['phase_symbol'].tolist()

    def format_meas_mineral_comp(self, mineral_comp, o_site_total):
        mineral_mol_elem_comp = self.mol_oxide_to_elem(mineral_comp)
        meas_elem_comp = mineral_mol_elem_comp*o_site_total/mineral_mol_elem_comp[-1]

        return meas_elem_comp

    def _validate_site_occ_input(self, single_value_index, site_id_index, endmember_site_occ,
                                 site_totals):

        for ival in single_value_index:
            index = site_id_index[ival]
            for iendmem in endmember_site_occ:
                assert iendmem[-1] ==site_totals[-1], (
                             'The last column of the site occupancy matrix and last element '
                             'of the site totals array must be equal. Recall, the last column '
                             'of the site_occupancy matrix must be oxygen and must have '
                             'values equal to the last element of the site totals array.')

                assert iendmem[index]==site_totals[ival], (
                             'Invariant sites in the site occ matrix are not equal to'
                             'the site totals.')

    def _success_test(self, residual, threshold):

        if np.max(np.abs(residual))< threshold:
            success = 'minimization successful'

        else:
            success = 'minimization failed; residual greater than threshold value'

        return success

    def lstsqr_endmember_comp(self, mol_oxide_comp, mol_oxide_comp_endmembers, decimals):
        #mol_oxide_comp_endmembers = phases.props['mol_oxide_comp']
        #mol_oxide_comp_endmembers = modelDB.phases[abbrev].props['mol_oxide_comp']

        output = np.linalg.lstsq(
            mol_oxide_comp_endmembers.T, mol_oxide_comp, rcond=None)

        endmember_comp = output[0]
        endmember_comp = np.round(endmember_comp, decimals=decimals)

        mol_oxide_comp_model = np.dot(mol_oxide_comp_endmembers.T, endmember_comp)
        mol_oxide_comp_residual = (mol_oxide_comp - mol_oxide_comp_model)

        return endmember_comp, mol_oxide_comp_model, mol_oxide_comp_residual

    def site_spec_lstsq_endmember_comp(self, meas_elem_comp, endmember_elem_stoic,
                             endmember_site_occ):
        """
        Infer endmember composition using least squares method.

        Parameters:
        ==========
        meas_elem_comp: array
            measured mole element composition for specific phase
            elements must be in the order (Si, Ti, Al, Fe, Cr, Mn,
            Mg, Ni, Co, Ca, Na, K, P, H, C, O))

        endmember_elem_stoic: array
            stoichiometry of endmembers in terms of elements
            rows = individual endmembers
            columns = elements (following the order Si, Ti, Al, Fe, Cr, Mn,
            Mg, Ni, Co, Ca, Na, K, P, H, C, O)

        endmember_site_occ: array
            site occupancies for each endmember
            rows = individual endmembers
            columns = site occupancies in the order of elements on X site,
            elements on Y site, T site, followed by oxygen.

        Returns:
        ========
        endmember_comp_lsq: np array
            endmember proportions from least squares fit

        site_occ_lsq: array
            site occupancy proportions from least squares fit

        resid_lsq: array
            least squares residual

        """
        endmember_comp_lsq, resid_lsq, rank, sing_vals = np.lingalg.lstsq(endmember_elem_stoic.T, meas_elem_comp)
        site_occ_lsq = np.dot(endmember_site_occ.T, endmember_comp_lsq)

        return endmember_comp_lsq, site_occ_lsq, resid_lsq

    def nnls_endmember_comp(self, meas_elem_comp, endmember_elem_stoic, endmember_site_occ):

        endmember_comp_nnls, resid_nnls = optimize.nnls(endmember_elem_stoic.T, meas_elem_comp)
        site_occ_nnls = np.dot(endmember_site_occ.T, endmember_comp_nnls)

        return endmember_comp_nnls, site_occ_nnls, resid_nnls


    def _null_vectors(self, endmember_site_occ_dev):
        u, s, vh = np.linalg.svd(endmember_site_occ_dev)

        null_vectors=[]
        threshold = 1e-10
        for ivh in vh:
            if np.all(np.abs((endmember_site_occ_dev.dot(ivh))) <= threshold):
                null_vectors.append(ivh)

            else:
                pass

        return null_vectors

    def _site_occ_constraints(self, endmember_comp_lsq, site_occ_stoic, meas_elem_comp,
                              endmember_site_occ, null_vectors, site_occ_lsq,
                              avg_endmember_site_occ, site_totals, site_id, uniq_site_id,
                              single_value_index):

        fn = lambda endmember_comp_lsq, site_occ_stoic, meas_elem_comp: (
                    np.linalg.norm(site_occ_stoic.dot(endmember_comp_lsq) - meas_elem_comp))

        bounds = [[0., None]]*len(endmember_site_occ.T)
        cons = []
        for inull_vec in null_vectors:
            con = {}
            con['type'] = 'eq'
            con['fun'] = lambda site_occ_lsq, avg_endmember_site_occ=avg_endmember_site_occ, inull_vec=inull_vec: (
                                inull_vec.dot(site_occ_lsq-avg_endmember_site_occ))

            cons.append(con)

        for ivalue in single_value_index:
            uniq_site_id.remove(ivalue)

        for isite in uniq_site_id:
            imask = site_id ==isite
            isite_dev = lambda site_occ_lsq, imask=imask, site_totals=site_totals,isite=isite: (
                                      np.sum(site_occ_lsq[imask])-site_totals[isite])
            cons.append({'type': 'eq', 'fun': isite_dev})

        return fn, bounds, cons

    def _constrained_minimization(self, fn, site_occ_lsq, site_occ_stoic, meas_elem_comp, bounds, cons,
                                    endmember_site_occ, threshold):

        sol = minimize(fn, site_occ_lsq, args=(site_occ_stoic, meas_elem_comp), method='SLSQP',
                       bounds=bounds, constraints=cons)

        site_occ_constr = sol.x

        endmember_comp_constr = np.linalg.pinv(endmember_site_occ.T).dot(site_occ_constr)
        resid_constr = site_occ_stoic.dot(site_occ_constr) - meas_elem_comp
        rms_resid_constr = np.sqrt(np.mean(resid_constr**2))

        success = self._success_test(resid_constr, threshold)

        return site_occ_constr, endmember_comp_constr, resid_constr, rms_resid_constr, success

    def infer_endmember_comp(self, meas_elem_comp, endmember_elem_stoic,
                             endmember_site_occ, site_totals, site_id,
                             threshold=1e-1, lstsq_fit=False, output=True):
        """
        Infer endmember composition using minimization method.

        Notes
        -----
        * This method is not currently being used in calibration code; there
        is an analogous function in phases.py called "calc_endmember_comp"

        * This function implements a complex mechanism of calculating endmember
        compositions in which it involves detailed site occupancy constraints
        not inherent in either the intrinsic or least squares minimizations to
        get endmember compositions

        * This function has a more extensive output and will spit out site
        occupanices under least squares and constrained minimizations as well
        as residuals and success statements

        Parameters:
        ==========
        meas_elem_comp: array
            measured mole element composition for specific phase

        endmember_elem_stoic: array
            stoichiometry of endmembers in terms of elements
            rows = individual endmembers
            columns = elements (following the order Si, Ti, Al, Fe, Cr, Mn,
            Mg, Ni, Co, Ca, Na, K, P, H, C, O)

        endmember_site_occ: array
            site occupancies for each endmember
            rows = individual endmembers
            columns = site occupancies in the order of elements on X site,
            elements on Y site, T site, followed by oxygen.

        site_totals: array
            total atoms allowed on each site

        output: True, default
            boolean flag indicating whether full output dictionary is returned;
            if False, function will return endmember proportions only

        Returns:
        ========
        output: dict
            dictionary contents are as follows:
                endmember_comp_lsq: array
                    endmember proportions from least squares fit

                site_occ_lsq: array
                    site occupancie proportions from least squares fit

                site_occ_constr: array
                    final site occupancy proportions from minimization function

                endmember_comp_constr: array
                    final endmember proportions after minimization

                resid_lsq: array
                    least squares residual

                resid_constr: array
                    residual using site proportions from minimization
                    (minimizes Ax-b=0)

                rms_resid_constr: int
                    root mean square of resid_constr array

        OR

        endmember_comp_constr: array
            final endmember proportions after minimization

        """
        site_id_info = np.unique(site_id, return_counts=True, return_index=True)
        uniq_site_id = site_id_info[0]
        uniq_site_id = uniq_site_id.tolist()
        site_id_index = site_id_info[1]
        site_id_count = site_id_info[2]

        single_value_index, = np.where(site_id_count==1)

        self._validate_site_occ_input(single_value_index, site_id_index, endmember_site_occ,
                                     site_totals)

        if lstsq_fit == True:

            endmember_comp_lsq, site_occ_lsq, resid_lsq = (
                self.lstsq_endmember_comp(meas_elem_comp, endmember_elem_stoic,
                                        endmember_site_occ))

            endmember_site_occ_inv = np.linalg.pinv(endmember_site_occ.T)
            site_occ_stoic = np.dot(endmember_elem_stoic.T, endmember_site_occ_inv)

            avg_endmember_site_occ = np.mean(endmember_site_occ, axis=0)
            endmember_site_occ_dev = endmember_site_occ-avg_endmember_site_occ

            null_vectors = self._null_vectors(endmember_site_occ_dev)

            fn, bounds, cons = self._site_occ_constraints(endmember_comp_lsq, site_occ_stoic, meas_elem_comp,
                                                     endmember_site_occ, null_vectors, site_occ_lsq,
                                                     avg_endmember_site_occ, site_totals, site_id, uniq_site_id,
                                                     single_value_index)

            site_occ_constr, endmember_comp_constr, resid_constr, rms_resid_constr, success = (
                self._constrained_minimization(fn, site_occ_lsq, site_occ_stoic,meas_elem_comp, bounds, cons,
                                               endmember_site_occ, threshold))

            if output == True:
                output = {}
                output['success'] = success
                output['endmember_comp_lsq'] = endmember_comp_lsq
                output['site_occ_lsq'] = site_occ_lsq
                output['resid_lsq'] = resid_lsq
                output['site_occ_constr'] = site_occ_constr
                output['endmember_comp_constr'] = endmember_comp_constr
                output['resid_constr'] = resid_constr
                output['rms_resid_constr'] = rms_resid_constr

                return output

            else:
                return success, endmember_comp_constr

        else:
            endmember_comp_nnls, site_occ_nnls, resid_nnls = (
                                self.nnls_endmember_comp(meas_elem_comp, endmember_elem_stoic,
                                                         endmember_site_occ))

            success = self._success_test(resid_nnls, threshold)

            if output == True:
                output = {}
                output['success'] = success
                output['endmember_comp_nnls'] = endmember_comp_nnls
                output['site_occ_nnls'] = site_occ_nnls
                output['resid_nnls'] = resid_nnls

                return output

            else:
                return success, endmember_comp_nnls

    def calc_reaction_svd(self, phase_symbols, TOLsvd=1e-4, modelDB=None):
        """
        Obtain svd of valid set of reactions

        Parameters
        ----------
        phase_symbols : list of strings
            list of phases (according to Berman abbreviations) that are
            allowed to participate in any given reaction
        TOL :

        Returns
        -------
        rxn_svd: array
            set of linearly independent reactions with variance minimized and
            orthogonality maximized
        """
        if modelDB is None:
            from thermoengine import model
            modelDB = model.Database()

        oxide_num = self.oxide_props['oxide_num']
        all_mol_oxide_comp = np.zeros((0,oxide_num))
        all_phase_name = []
        all_phase_symbol = []
        all_endmember_name = []
        all_endmember_id = np.zeros((0))
        all_phase_ind = np.zeros((0))
        all_atom_num = []

        for (ind_phs, iabbrev) in enumerate(phase_symbols):
            #iphs_props = modelDB.phase_attributes[iabbrev]['props']
            iphs_props = modelDB.phases[iabbrev].props
            iall_mol_oxide_comp = iphs_props['mol_oxide_comp']
            iphase_name = iphs_props['phase_name']
            iendmember_name = iphs_props['endmember_name']
            iendmember_num = len(iendmember_name)
            iendmember_id = np.arange(iendmember_num)
            iatom_num = iphs_props['atom_num']

            #iphase_name_tile = np.tile(np.array([iphs_props['phase_name']]), iendmember_num)
            all_phase_ind = np.hstack((all_phase_ind, np.tile(ind_phs,(iendmember_num))))
            all_mol_oxide_comp= np.vstack((all_mol_oxide_comp, iall_mol_oxide_comp))
            all_phase_name.extend([iphase_name for i in range(iendmember_num)])
            all_phase_symbol.extend([iabbrev for i in range(iendmember_num)])
            all_endmember_name.extend(iendmember_name)
            all_endmember_id = np.hstack((all_endmember_id, iendmember_id))
            all_atom_num.extend(iatom_num)


        endmember_num = len(all_endmember_name)
        all_atom_num = np.array(all_atom_num)
        # oxide_atom_num = self.oxide_props['cat_num']+self.oxide_props['oxy_num']
        # oxide_comp_per_atom = all_mol_oxide_comp/np.tile(oxide_atom_num[np.newaxis,:],(endmember_num,1))
        # oxide_comp_per_atom /= np.tile(np.sum(oxide_comp_per_atom, axis=1)[:,np.newaxis],(1,oxide_atom_num.size))
        # np.sum(oxide_comp_per_atom, axis=1)
        # Nendmember, Noxide = oxide_comp_per_atom.shape
        # u, s, vh =np.linalg.svd(oxide_comp_per_atom.T)

        # convert all_mol_oxide comp to moles of atoms for each oxide; divide by oxide num and norm
        # to 1;
        Nendmember, Noxide = all_mol_oxide_comp.shape
        u, s, vh =np.linalg.svd(all_mol_oxide_comp.T)


        TOL = TOLsvd
        N_nonrxn = np.sum(np.abs(s)>= TOL)
        N_rxn = Nendmember-Noxide + np.sum(np.abs(s)< TOL)
        non_rxn = vh[0:N_nonrxn]
        rxn_svd0 = vh[N_nonrxn:]

        scl = np.array([np.linalg.norm(irxn) for irxn in rxn_svd0])
        rxn_svd0 = rxn_svd0/np.tile(scl[:,np.newaxis],(1,endmember_num))

        # Describe code below; add to documentation
        reac_atom_num = 0.5*np.sum(np.abs(rxn_svd0)*all_atom_num, axis=1)
        rxn_svd = rxn_svd0*all_atom_num[np.newaxis,:]/(
            reac_atom_num[:,np.newaxis])

        rxn_svd_props = OrderedDict()
        rxn_svd_props['rxn_svd'] = rxn_svd
        rxn_svd_props['oxide_num'] = oxide_num
        rxn_svd_props['all_mol_oxide_comp'] = all_mol_oxide_comp
        rxn_svd_props['all_phase_name'] =  all_phase_name
        rxn_svd_props['all_phase_symbol'] = all_phase_symbol
        rxn_svd_props['all_endmember_name'] =  all_endmember_name
        rxn_svd_props['all_endmember_id'] = all_endmember_id
        rxn_svd_props['all_phase_ind'] = all_phase_ind
        rxn_svd_props['all_atom_num'] = all_atom_num
        #self._rxn_svd_props = rxn_svd_props #TK: What is the purpose of this?

        return rxn_svd_props

    #rxn_svd, rxn_svd_props = get_reaction_svd(phase_symbols, TOLsvd=1e-4)

    def get_wtcoefs_ortho(self, wtcoefs, wtcoefs_ortho, apply_norm=True):
        wts_ortho = wtcoefs.copy()
        for iwtcoefs_ortho in wtcoefs_ortho:
            wts_ortho -= np.dot(iwtcoefs_ortho, wts_ortho)*iwtcoefs_ortho

        if apply_norm:
            wts_ortho /= np.linalg.norm(wts_ortho)

        return wts_ortho

    def random_rxn(self, wtcoefs_ortho=None, Nbasis=36):
        wts = 2*np.random.rand(Nbasis)-1
        wts /= np.linalg.norm(wts)

        if wtcoefs_ortho is not None:
            wts = self.get_wtcoefs_ortho(wts, wtcoefs_ortho)

        return wts

    def _ortho_penalty(self, wts, wtcoefs_ortho=None, scale=1):
        if wtcoefs_ortho is None:
            cost = 0
        else:
            wts_ortho = self.get_wtcoefs_ortho(wts, wtcoefs_ortho, apply_norm=False)
            frac = np.minimum(np.linalg.norm(wts_ortho)/np.linalg.norm(wts), 1)
            cost = scale*(1-frac**2)

        return cost

    def rxn_complexity(self, rxn_svd, TOL=1e-10):

        irxn_abs = np.abs(rxn_svd)
        xlogx = irxn_abs*np.log(irxn_abs)
        xlogx[irxn_abs<TOL] = 0
        complexity = -np.sum(xlogx)

        return complexity

    def lasso_rxn_complexity(self, rxn_svd, scl=1.0):
        irxn_abs = np.abs(rxn_svd)
        complexity = -np.sum(-irxn_abs/scl-np.log(2*scl))
        return complexity

    def linear_combo(self, wts, rxn, return_scl=False):
        wts = np.array(wts)
        endmember_num = rxn.shape[1]
        scl = np.tile(wts[:,np.newaxis],(1,endmember_num))
        rxn_wt = np.sum(scl*rxn, axis=0)
        scl_tot = np.linalg.norm(rxn_wt)
        rxn_wt /= scl_tot

        if return_scl:
            return rxn_wt, wts/scl_tot

        else:
            return rxn_wt
    def rxn_costfun(self, wts, rxn, wtcoefs_ortho=None, ortho_scale=1, lasso_scale=1, debug=False, TOL=1e-10):
        rxn_wt = self.linear_combo(wts, rxn)
        # complexity = self.rxn_complexity(rxn_wt, TOL=TOL)
        complexity = self.lasso_rxn_complexity(
            rxn_wt, scl=lasso_scale)
        cost = complexity + self._ortho_penalty(wts, wtcoefs_ortho=wtcoefs_ortho, scale=ortho_scale)

        if debug:
            if wtcoefs_ortho is not None:
                wts_ortho = self.get_wtcoefs_ortho(wts, wtcoefs_ortho, apply_norm=False)
                print('norm(wts) = {wts_norm}'.format(wts_norm=np.linalg.norm(wts)))
                print('norm(ortho) = {ortho_norm}'.format(ortho_norm=np.linalg.norm(wts_ortho)))
                ortho_cost = self._ortho_penalty(wts, wtcoefs_ortho=wtcoefs_ortho, scale=ortho_scale)
                print('ortho_cost = ', ortho_cost)

        return cost


    def _draw_basic_rxns(self, rxn_svd, wtcoefs_ortho=None, Ndraw=10, ortho_scale=1, lasso_scale=1.0, Nbasis=36, TOL=1e-10):
        from scipy import optimize
        Nbasis = rxn_svd.shape[0]
        Nendmem = rxn_svd.shape[1]

        wtcoefs = np.zeros((Ndraw, Nbasis))
        cost = np.zeros(Ndraw)
        rxn_coefs = np.zeros((Ndraw, Nendmem))

        for ind in range(Ndraw):
            iwts0 = self.random_rxn(wtcoefs_ortho=wtcoefs_ortho, Nbasis=Nbasis)

            def costfun(wts, rxn=rxn_svd, wtcoefs_ortho=wtcoefs_ortho,lasso_scale=lasso_scale, ortho_scale=ortho_scale):
                return self.rxn_costfun(wts, rxn, wtcoefs_ortho=wtcoefs_ortho, lasso_scale=lasso_scale, ortho_scale=ortho_scale, TOL=TOL)

            for ind_fit in range(1):

                ifit = optimize.minimize(costfun, iwts0, tol=1e-10)
                iwts0 = ifit['x']


            iwt_fit = ifit['x']
            self.rxn_costfun(iwt_fit, rxn_svd, wtcoefs_ortho=wtcoefs_ortho, lasso_scale=lasso_scale, ortho_scale=ortho_scale, debug=True, TOL=TOL)

            irxn_coefs = self.linear_combo(iwt_fit, rxn_svd)
            wtcoefs[ind] = iwt_fit
            rxn_coefs[ind] = irxn_coefs
            cost[ind] = ifit['fun']

        return wtcoefs, rxn_coefs, cost

    def next_basic_rxn(self, rxn_svd, wtcoefs_ortho=None, Ndraw=10, ortho_scale=1, lasso_scale=1, Nbasis=36, TOL=1e-10):
        wtcoefs, rxn_coefs, cost = self._draw_basic_rxns(rxn_svd, wtcoefs_ortho=wtcoefs_ortho, Ndraw=Ndraw, ortho_scale=ortho_scale,
                                                         lasso_scale=lasso_scale, Nbasis=Nbasis, TOL=TOL)

        ind = np.argmin(cost)

        return wtcoefs[ind], rxn_coefs[ind], cost[ind]


    def get_rxns(self, rxn_svd, Ndraw=10, ortho_scale=10, lasso_scale=1.0, Nbasis=36, TOL=1e-10):
        """
        filter rxn_svd based on cost analysis and get basic reactions

        Parameters
        ----------

        rxn_svd : double array
            set of linearly independent reactions


        Returns
        -------


        """

        Nbasis = rxn_svd.shape[0]
        Nendmem = rxn_svd.shape[1]


        wtcoefs_ortho = np.zeros((Nbasis, Nbasis))
        wtcoefs = np.zeros((Nbasis, Nbasis))
        rxn_coefs = np.zeros((Nbasis, Nendmem))
        costs = np.zeros(Nbasis)

        for ind in range(Nbasis):
            iwtcoefs, irxn_coefs, icost = self.next_basic_rxn(rxn_svd, wtcoefs_ortho=wtcoefs_ortho, Ndraw=Ndraw, ortho_scale=ortho_scale, lasso_scale=lasso_scale, Nbasis=Nbasis, TOL=TOL)

            iwtcoefs_ortho = self.get_wtcoefs_ortho(iwtcoefs, wtcoefs_ortho)

            wtcoefs[ind] = iwtcoefs
            wtcoefs_ortho[ind] = iwtcoefs_ortho
            rxn_coefs[ind] = irxn_coefs
            costs[ind] = icost
            print('icost({ind}) = {icost}'.format(ind=ind,icost=icost))
            print('=====')
        return wtcoefs, costs, rxn_coefs, wtcoefs_ortho




chem = _Chem()


class UnorderedList(collections.UserList):
    def __eq__(self, other):
        return sorted(self.data) == sorted(other)