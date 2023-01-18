"""
This module provides Python wrappers to individual and collections of phases.

"""

from thermoengine import core
from thermoengine import chem
import numpy as np
import pandas as pd
from os import path
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from importlib import import_module, invalidate_caches
import sys
import deprecation
from functools import wraps

# specialized numerical imports
from scipy import optimize as optim

from typing import Type, List

# __all__ = ['Rxn','Assemblage','PurePhase','SolutionPhase','get_phaselist']
__all__ = ['Rxn','Assemblage','Phase','PurePhase','SolutionPhase','get_phaselist']

DATADIR = 'data/phases'
PURE_PHASE_FILE = 'PurePhaseList.csv'
SOLUTION_PHASE_FILE = 'SolutionPhaseList.csv'

# inputs
#
# * P=1, T=1, mol=1
# * P=N_PTX, T=1, mol=1
# * P=1, T=N_PTX, mol=1
# * P=1, T=1, mol = N_PTX
# * P=N_PTX, T=N_PTX, mol=N_PTX
#
# qty
#
# * N_endmem (deriv_type)
# * N_rxn
# * N_phase
#
# outputs
#
# * N_PTX,
# * N_PTX, N_endmem
# * N_PTX, N_endmem, N_endmem
#
# * N_phase,
# * N_rxn, N_phase
# * N_rxn, N_phase, N_PTX
#
# * reduce dimensions if N_PTX=1
#===================================================
def get_phase_info():
    """
    Get acceptable table of basic phase info
        (e.g., name, abbrev, formula, members, etc.).

    Returns
    -------
    phase_table : dict
        Dictionary containing phase information.
            - 'purephases' : pandas dataframe
            - 'purefilenm' : str defining source file for pure phase info

    """

    phase_info = {}
    phase_info['pure'] = _read_phase_info(PURE_PHASE_FILE)
    phase_info['solution'] = _read_phase_info(SOLUTION_PHASE_FILE)

    info_files = {}
    info_files['pure'] = PURE_PHASE_FILE
    info_files['solution'] = SOLUTION_PHASE_FILE
    return phase_info, info_files

def _read_phase_info(filenm):
    """
    Read phase info tables

    Internal method to read phase abbreviations and names from csv files.

    """

    parentpath = path.dirname(__file__)
    pathname = path.join(parentpath,DATADIR,filenm)
    try:
       phases_info_df = pd.read_csv(pathname)
    except:
        assert False,'The '+pathname+' file cannot be found. '\
            'It is needed to define the standard phase abbreviations.'

    return phases_info_df
#===================================================
class FixedRxnSet:
    def __init__(self, phase_symbols, endmember_ids, rxn_coefs,
                 phases_dict, T, P, mols,):

        self._init_rxn_set(phase_symbols, endmember_ids, rxn_coefs,
                           phases_dict)
        self._init_exp_cond(T, P, mols)

    def _init_rxn_set(self, phase_symbols, endmember_ids, rxn_coefs,
                      phases_dict):

        assert np.all(
            [sym in phases_dict.keys() for sym in phase_symbols ]), (
                'phase symbols must appear in phase_obj_dict'
            )

        self._phase_symbols = phase_symbols
        self._endmember_ids = endmember_ids
        self._rxn_coefs = rxn_coefs
        self._phases_dict = phases_dict
        self._rxn_num = rxn_coefs.shape[0]
        self._endmember_num = rxn_coefs.shape[1]

    def _init_exp_cond(self, T, P, mols):
        phase_symbols = self._phase_symbols

        # validate mols
        self._validate_mol_input(mols)

        assert np.all(
            [sym in phase_symbols for sym in mols.keys()]),(
                'molar composition must be defines for every phase'
            )

        self._T = T
        self._P = P
        self._mols = mols

    def _validate_mol_input(self, mols):
        mols = {} if mols is None else mols

        phases = self.phases
        phase_symbols = self.phase_symbols

        for symbol in phase_symbols:
            if symbol not in mols:
                mols[symbol] = None

        assert np.all(np.array(
            [symbol in phase_symbols for symbol in mols.keys()])), (
                'Invalid phase symbol(s) used to define phase '
                'composition in mols. Only use valid phase symbols '
                'found in rxn.phase_symbols.'
            )

        return mols

    def affinity(self):
        rxn_coefs = self._rxn_coefs

        phase_chem_potentials = self.phase_chem_potentials()
        chem_potentials = self.endmem_chem_potentials(
            phase_chem_potentials)

        affinities = np.dot(rxn_coefs, chem_potentials)
        return affinities

    def endmem_chem_potentials(self, phase_chem_potentials):
        endmember_num = self._endmember_num
        phase_symbols = self._phase_symbols
        endmember_ids = self._endmember_ids

        chem_potentials = np.zeros(endmember_num)
        for ind, (phs_sym, endmem_id) in enumerate(
            zip(phase_symbols, endmember_ids)):
            chem_potentials[ind] = phase_chem_potentials[phs_sym][endmem_id]

        return chem_potentials

    def phase_chem_potentials(self):
        phases_dict = self._phases_dict

        T = self._T
        P = self._P
        mols = self._mols

        phase_chem_potentials = {}

        for phs_sym in phases_dict:
            phs = phases_dict[phs_sym]
            phase_chem_potentials[phs_sym] = phs.chem_pot(T, P, mols=mols)

        return phase_chem_potentials




#===================================================
class Rxn:
    """
    Class that defines identity/properties of a specific phase reaction.

    Reactions occur between phases (either pure or solution) and are
    defined in terms of the participating endmembers, indicating which atoms
    are exchanged between phases during the reaction.

    Parameters
    ----------
    phase_objs : array of Phase Objects
        Defines which phases participate in the reaction.
    endmember_ids : int array
        Indicates the endmember of each phase that participates in the reaction.
        This array must have the same order as the phase array (phase_objs).
    rxn_coefs : double array
        Defines the stoichiometric rxn coefficient, where negative values
        are reactants and positive values are products. The reaction must be
        balanced (obeying mass conservation).
        This array must have the same order as the phase array (phase_objs).

    Attributes
    ----------
    endmember_ids
    endmember_names
    phase_num
    phase_symbols
    phases
    product_phases
    reactant_phases
    rxn_coefs

    Notes
    -----
    * The phases themselves may be pure or have realistic
      intermediate compositions (if they are solution phases).
    * The reaction is defined in terms of the exchange of endmembers between
      the participating phases.
    * Reaction coefficients correspond to a balanced stoichiometric reaction.

    """
    def __init__(self, phase_objs, endmember_ids, rxn_coefs,
                 coefs_per_atom=False):
        self._validate_inputs(phase_objs, endmember_ids, rxn_coefs)
        self._init_rxn(phase_objs, endmember_ids, rxn_coefs,
                              coefs_per_atom)

        # TK
        # self._validate_rxn_balance

        pass

    def _validate_inputs(self, phase_objs, endmember_ids, rxn_coefs):
        Nphase = len(phase_objs)
        rxn_coefs = np.array(rxn_coefs)

        assert len(endmember_ids) == Nphase, (
            'endmember_ids must provide endmember index for '
            'every reaction phase. '
            )
        assert len(rxn_coefs) == Nphase, (
            'rxn_coefs must provide stoichiometric '
            'coefficients for every reaction phase.'
            )

        assert np.any(rxn_coefs<0), (
            'rxn_coefs must indicate reactants with negative '
            'sign and products with positive sign. Currently, '
            'none of the stoichiometric coefficients are negative.'
            )
        assert np.any(rxn_coefs>0), (
            'rxn_coefs must indicate reactants with negative '
            'sign and products with positive sign. Currently, '
            'none of the stoichiometric coefficients are positive.'
            )

    def _init_rxn(self, phase_objs, endmember_ids, rxn_coefs, coefs_per_atom):
        phase_objs, endmember_ids, rxn_coefs = (
            self._trim_absent_rxn_phases(
                phase_objs, endmember_ids, rxn_coefs))

        phase_objs = np.array(phase_objs)
        endmember_ids = np.array(endmember_ids)

        Nphase = len(phase_objs)
        rxn_coefs = np.array(rxn_coefs)

        all_atom_nums = []
        for phs, endmember_id in zip(phase_objs, endmember_ids):
            endmember_id = int(endmember_id)
            iatom_num = phs.props['atom_num'][endmember_id]
            all_atom_nums.append(iatom_num)
        all_atom_nums = np.array(all_atom_nums)

        if coefs_per_atom:
            rxn_coefs = rxn_coefs/all_atom_nums

        phase_symbols = []
        endmember_names = []


        for phase_obj, iendmember_id in zip(phase_objs, endmember_ids):
            phase_symbols.append(
                phase_obj.props['abbrev'])

            endmember_names.append(
                phase_obj.endmember_names[int(iendmember_id)])



        phase_symbols = np.array(phase_symbols)

        # indsort = np.argsort(phase_symbols)
        # self._set_phase_assemblage(
        #     phase_objs[indsort], phase_symbols[indsort],
        #     obj_is_classnm=obj_is_classnm)

        self._phase_num = Nphase
        self._phase_symbols = phase_symbols
        self._phases = phase_objs
        self._endmember_ids = endmember_ids
        self._endmember_names = endmember_names
        self._rxn_coefs = rxn_coefs

    def _trim_absent_rxn_phases(self, phase_objs, endmember_ids, rxn_coefs):
        remove_set, = np.where(np.atleast_1d(rxn_coefs) == 0)

        # re-create lists for items not in remove set
        rxn_coefs = [v for i, v in enumerate(rxn_coefs)
            if i not in remove_set]
        phase_objs = [v for i, v in enumerate(phase_objs)
            if i not in remove_set]
        endmember_ids = [v for i, v in enumerate(endmember_ids)
            if i not in remove_set]

        return phase_objs, endmember_ids, rxn_coefs

    @property
    def phase_num(self):
        """
        Number of phases

        Returns
        -------
        Number of phases (int)

        """
        return self._phase_num

    @property
    def phases(self):
        """
        Phase objects used in the reaction

        Returns
        -------
        Array of phase objects used in the reaction

        """
        return self._phases

    @property
    def reactant_phases(self):
        """
        Reactant phases

        Returns
        -------
        Array of reactant phase objects

        """
        return self.phases[self.rxn_coefs<0]

    @property
    def product_phases(self):
        """
        Product phases

        Returns
        -------
        Array of product phase objects

        """
        return self.phases[self.rxn_coefs>0]

    @property
    def phase_symbols(self):
        """
        Phase symbols

        Returns
        -------
        Array of phase symbols used in the reaction (str)

        """
        return self._phase_symbols

    @property
    def endmember_ids(self):
        """
        ID number of each endmember in phase

        Returns
        -------
        Array of ids, [int,...]

        """
        return self._endmember_ids

    @property
    def endmember_names(self):
        """
        Name of each endmember

        Returns
        -------
        List of endmember names for this solution phase, [str,...]

        """
        return self._endmember_names

    @property
    def rxn_coefs(self):
        """
        Reaction coefficients

        Returns
        -------
        Array of reaction coefficients (double)

        """
        return self._rxn_coefs

    def _validate_mol_input(self, mols):
        mols = {} if mols is None else mols

        phases = self.phases
        phase_symbols = self.phase_symbols

        for symbol in phase_symbols:
            if symbol not in mols:
                mols[symbol] = None

        assert np.all(np.array(
            [symbol in phase_symbols for symbol in mols.keys()])), (
                'Invalid phase symbol(s) used to define phase '
                'composition in mols. Only use valid phase symbols '
                'found in rxn.phase_symbols.'
            )

        return mols

    def affinity(self, T, P, mols=None):
        """
        Calculate reaction affinity

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin
        P : array-like
            Pressure in bars
        mol : dict of arrays, optional
            Composition of each phase in terms of mols of endmembers
            (unneeded for pure phases)

        Returns
        -------
        value : array-like
            Reaction affinity in J

        """
        affinity = -self.chem_potential(T, P, mols=mols)
        return affinity

    def chem_potential(self, T, P, mols=None):
        """
        Calculate net chemical potential change of the reaction

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin
        P : array-like
            Pressure in bars
        mol : dict of arrays, optional
            Composition of each phase in terms of mols of endmembers
            (unneeded for pure phases)

        Returns
        -------
        value : array-like
            Chemical potential in J for the net change of the reaction

        """

        chem_potential = self._calc_net_rxn_values(
            'chem_potential', T, P, mols=mols, use_endmember=True)
        return chem_potential

    def volume(self, T, P, mols=None, peratom=False):
        """
        Calculate net volume change of the reaction

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin
        P : array-like
            Pressure in bars
        mol : dict of arrays, optional
            Composition of each phase in terms of mols of endmembers
            (unneeded for pure phases)

        Returns
        -------
        value : array-like
            Volume in J/bar for the net change of the reaction

        """

        rxn_volume = self._calc_net_rxn_values('volume', T, P, mols=mols,
                                               peratom=peratom)
        return rxn_volume

    def entropy(self, T, P, mols=None, peratom=False):
        """
        Calculate net entropy change of the reaction

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin
        P : array-like
            Pressure in bars
        mol : dict of arrays, optional
            Composition of each phase in terms of mols of endmembers
            (unneeded for pure phases).

        Returns
        -------
        value : array-like
            Entropy in J/K for the net change of the reaction.

        """

        rxn_entropy = self._calc_net_rxn_values('entropy', T, P, mols=mols,
                                               peratom=peratom)
        return rxn_entropy

    def boundary(self, T=None, P=None, mols=None, init_guess=None):
        Tstd = 300.0
        Pstd = 1.0

        assert (T is not None) or (P is not None), \
            'Both T and P are None; Must define either temperature or pressure.'

        if T is not None:
            if init_guess is None:
                init_guess = Pstd

            Afun = lambda P, T=T: self.affinity(T, P, mols=mols)
            dAfun = lambda P, T=T: -self.volume(T, P, mols=mols, peratom=True)

        else:
            if init_guess is None:
                init_guess = Tstd

            Afun = lambda T, P=P: self.affinity(T, P, mols=mols)
            dAfun = lambda T, P=P: +self.entropy(T, P, mols=mols, peratom=True)

        value_bound = optim.newton(Afun, init_guess, fprime=dAfun)

        return value_bound

    def clapeyron_slope(self, T, P, mols=None, peratom=False):
        dV_rxn = self.volume(T, P, mols=mols, peratom=peratom)
        dS_rxn = self.entropy(T, P, mols=mols, peratom=peratom)
        dTdP = dV_rxn / dS_rxn
        return dTdP

    def simultaneous_rxn_cond(self, other_rxn, Pinit=1.0, TOL=1e-5):
        P = Pinit

        while True:
            Tbnd1 = self.boundary(P=P)
            Tbnd2 = other_rxn.boundary(P=P)

            dTdP1 = self.clapeyron_slope(Tbnd1,P)
            dTdP2 = other_rxn.clapeyron_slope(Tbnd2,Pinit)

            if np.abs(np.log(Tbnd1/Tbnd2)) < TOL:
                T =  0.5*(Tbnd1+Tbnd2)
                break

            dP = - (Tbnd1 - Tbnd2)/(dTdP1-dTdP2)
            P += dP

        T = float(T)
        P = float(P)

        return T, P

    def trace_boundary(self, Tlims=None, Plims=None, init_guess=None,
                       Nsamp=30):

        assert (Tlims is not None) or (Plims is not None), \
            'Both T and P are None; Must define either temperature or pressure.'

        if Tlims is not None:
            xvals = np.linspace(Tlims[0], Tlims[1], Nsamp)

            bound_fun = lambda T, init_guess: self.boundary(
                T=T, init_guess=init_guess)

            dAdx_fun = lambda x, y: +self.entropy(x, y)
            dAdy_fun = lambda x, y: -self.volume(x, y)

        else:
            xvals = np.linspace(Plims[0], Plims[1], Nsamp)

            bound_fun = lambda P, init_guess: self.boundary(
                P=P, init_guess=init_guess)

            dAdx_fun = lambda x, y: -self.volume(y, x)
            dAdy_fun = lambda x, y: +self.entropy(y, x)

        dx = xvals[1]-xvals[0]
        yvals = np.zeros(xvals.shape)

        for ind, x in enumerate(xvals):
            y = bound_fun(x, init_guess)
            dy_guess = dx*dAdx_fun(x=x, y=y)/dAdy_fun(x=x, y=y)
            yvals[ind] = y
            init_guess = y + dy_guess


        if Tlims is not None:
            T_bnds, P_bnds = xvals, yvals
        else:
            T_bnds, P_bnds = yvals, xvals

        return T_bnds, P_bnds

    def competing_rxn(self, T_bounds, P_bounds, other_rxn, TOL=1e-6):
        competing = False
        A_rxn_other = other_rxn.affinity(T_bounds, P_bounds)

        is_competing = np.tile(False, len(T_bounds))

        if np.all(other_rxn.reactant_phases in self.phases):
            is_competing[A_rxn_other>TOL] = True

        if np.all(other_rxn.product_phases in self.phases):
            is_competing[A_rxn_other<-TOL] = True

        return is_competing

    def stability(self, T_bounds, P_bounds, other_rxns, TOL=1e-6):
        # NOTE: other_rxns must be a list of reactions of the same composition
        peratom=True
        peratom=False

        # self._reac_assemblage
        # A_rxn_vals = self.affinity(T_bounds, P_bounds, peratom=peratom)
        A_rxn_vals = self.affinity(T_bounds, P_bounds)

        assert np.all(np.abs(A_rxn_vals)<TOL), \
            'Rxn affinity must be equal to zero to within TOL.'

        N_TP = A_rxn_vals.size
        N_rxn = len(other_rxns)

        # A_rxn_others = np.zeros((N_rxn, N_TP))

        # Calculate gibbs energy of rxn for all possible reactions

        stable = np.tile(True, N_TP)
        for ind, irxn in enumerate(other_rxns):
            is_competing = self.competing_rxn(T_bounds, P_bounds, irxn)

            stable[is_competing] = False

            # iA_rxn_other = irxn.affinity(T_bounds, P_bounds, peratom=peratom )
            # # if other rxn phase assemblage is a subset of the current
            # # assemblage, then it is a valid competing reaction

            # if self.competing_rxn(irxn):

            # if (rxn._reac_assemblage.issubset(self._reac_assemblage) | \
            #     rxn._reac_assemblage.issubset(self._prod_assemblage) ):
            #     G_rxn_other_a[ind] = iG_rxn_other_a

            # elif (rxn._prod_assemblage.issubset(self._reac_assemblage) | \
            #       rxn._prod_assemblage.issubset(self._prod_assemblage) ):
            #     # Store negative energy as reverse reaction energy change
            #     G_rxn_other_a[ind] = -iG_rxn_other_a

            # else:
            #     G_rxn_other_a[ind] = 0.0

        # print(G_rxn_other_a)


        # If any rxn is energetically favored, then current reaction is not
        # stable
        # stable_a = ~np.any(G_rxn_other_a < -TOL, axis=0)
        return stable

    def _validate_state_input(self, T, P, mols):
        N_T = np.asarray(T).size
        N_P = np.asarray(P).size
        N_solution_phases = len(mols)
        N_X = np.ones(N_solution_phases, dtype=int)

        for ind, iphs in enumerate(mols):
            imol = mols[iphs]
            if imol is None:
                N_X[ind] = 0
            else:
                imol = np.asarray(imol)
                if imol.ndim==1:
                    N_X[ind] = 1
                else:
                    N_X[ind] = imol.shape[0]

        assert np.all(N_X<=1), (
            'Multiple compositions not currently supported.'
        )

        N_all = np.hstack((N_T, N_P, N_X))
        N_PTX = np.max(N_all)

        assert np.all(np.isin(N_all, [0, 1, N_PTX])),(
            'The number of state points for T, P, mol must be compatible.'
        )

        state = {}
        state['N_T'] = N_T
        state['N_P'] = N_P
        state['N_X'] = N_X
        state['N_PTX'] = N_PTX

        # from IPython import embed;embed();import ipdb as pdb;pdb.set_trace()
        T = np.tile(np.asarray(T), N_PTX) if N_T < N_PTX else T
        P = np.tile(np.asarray(P), N_PTX) if N_P < N_PTX else P

        state['T'] = T
        state['P'] = P
        state['mols'] = mols
        return state

    def _calc_net_rxn_values(self, method_name, T, P, mols=None,
                             peratom=False, use_endmember=False):
        mols = self._validate_mol_input(mols)
        phase_num = self.phase_num
        endmember_ids = self.endmember_ids
        rxn_coefs = self.rxn_coefs

        state = self._validate_state_input(T, P, mols)
        T, P, mols, N_PTX = (state.get(key) for
                             key in ['T', 'P', 'mols', 'N_PTX'])
        values = np.zeros((phase_num, N_PTX))
        for i, (iphs, iendmember) in enumerate(zip(
            self.phases, endmember_ids)):

            iphs_abbrev = iphs.abbrev
            imol = mols[iphs_abbrev]
            iphs_method = getattr(iphs, method_name)
            if use_endmember:
                ival = iphs_method(T, P, mol=imol, endmember=iendmember)
            else:
                ival = iphs_method(T, P, mol=imol)

            # if peratom:
            #     ival /= self.rxn_atomnum

            # from IPython import embed;embed();import ipdb as pdb;pdb.set_trace()
            values[i] = ival

        net_rxn_values = np.dot(rxn_coefs, values)

        #from IPython import embed;embed();import pdb as pdb;pdb.set_trace()

        return net_rxn_values

    ######################
    #     NOT UPDATED    #
    ######################
    def gibbs_energy( self, T_a, P_a, peratom=False ):
        dG_rxn_a = self._calc_rxn_change('gibbs_energy_all', T_a, P_a,
                                         peratom=peratom)
        return dG_rxn_a

    def enthalpy( self, T_a, P_a, peratom=False ):
        dH_rxn_a = self._calc_rxn_change('enthalpy_all', T_a, P_a,
                                         peratom=peratom )
        return dH_rxn_a


    def _calc_reac_value( self, method_name, T_a, P_a, peratom=False ):
        reac_method = getattr(self.reac_assemblage, method_name)
        val_phs_a = reac_method( T_a, P_a )
        val_a = np.dot(self.reac_rxn_coef_a, val_phs_a)

        # if peratom:
        #     val_a /= self.rxn_atomnum

        return val_a

    def _calc_prod_value( self, method_name, T_a, P_a, peratom=False ):
        #prod_method = getattr(self.prod_assemblage, method_name)
        prod_method = getattr(self.product_phases, method_name)
        val_phs_a = prod_method( T_a, P_a )
        val_a = np.dot(self.prod_rxn_coef_a, val_phs_a)

        #
        #  if peratom:
        # #
        #   val_a /= self.rxn_atomnum

        return val_a

    def _calc_rxn_change( self, method_name, T_a, P_a, peratom=False ):
        val_prod_a = self._calc_prod_value( method_name, T_a, P_a, peratom=peratom )
        val_reac_a = self._calc_reac_value( method_name, T_a, P_a, peratom=peratom )
        val_rxn_a = val_prod_a-val_reac_a
        return val_rxn_a
#===================================================
class Assemblage:
    def __init__(self, phase_objs, obj_is_classnm=False):
        # Get phase symbol list
        phase_symbols = []
        for phase_obj in phase_objs:
            phase_symbols.append(
                phase_obj.props['abbrev'])

        phase_objs = np.array(phase_objs)
        phase_symbols = np.array(phase_symbols)

        indsort = np.argsort(phase_symbols)
        self._set_phase_assemblage(
            phase_objs[indsort], phase_symbols[indsort],
            obj_is_classnm=obj_is_classnm)
        pass

    def __eq__(self, other):
        return self._phases == other._phases

    def __lt__(self, other):
        if len(self._phases) < len(other._phases):
            return True

        return self._phases[0] < other._phases[0]

    def __gt__(self, other):
        if len(self._phases) > len(other._phases):
            return True

        return self._phases[0] > other._phases[0]

    def issubset(self, other):
        is_member = [phase in other._phases for phase in self._phases]
        return np.all(is_member)

    def _set_phase_assemblage(self, phase_objs,
                             phase_symbols,
                             obj_is_classnm=False):
        if obj_is_classnm:
            phase_classnms = phase_objs
            phase_objs = [
                PurePhase(phase_classnm, phasesym)
                for (phase_classnm, phasesym) in
                zip(phase_classnms, phase_symbols)]

        self._phase_symbols = phase_symbols
        self._phases = phase_objs

        props = {}
        props['formula_all'] = [phs.props['formula']
                                for phs in phase_objs]
        props['name_all'] = [phs.props['phase_name']
                             for phs in phase_objs]
        props['abbrev_all'] = [phs.props['abbrev']
                               for phs in phase_objs ]
        props['molwt_all'] = np.array([
            phs.props['molwt'] for phs in phase_objs])
        props['element_comp_all'] = np.vstack([
            phs.props['element_comp'] for phs
            in phase_objs])
        # NOTE replace symbols?
        # props['element_symbols_all'] = phase_objs[0].props['element_symbols']

        self._props = props

        pass

    def _validate_mol_input(self, mols):
        phases = self.phases
        phase_symbols = self.phase_symbols

        for symbol in phase_symbols:
            if symbol not in mols:
                mols[symbol] = None

            # else:
            #     imol = mols[symbol]
            #     iphase = phases[symbol]
            #     iprops = iphase.props
            #     iendmember_names = iprops['endmember_name']
            #     iendmember_num = len(iendmember_names)

        return mols

    @property
    def phase_symbols(self):
        return self._phase_symbols

    @property
    def phases(self):
        return self._phases

    @property
    def props(self):
        return self._props

    def get_endmember_comp_matrix(self):

        oxide_num = chem.oxide_props['oxide_num']
        all_mol_oxide_comp = np.zeros((0,oxide_num))
        all_phase_name = np.zeros((0))
        all_endmember_name = np.zeros((0))
        all_endmember_ind = np.zeros((0))
        all_phase_ind = np.zeros((0))

        ind_phs = 0
        for iphase in self.phases:
            iphs_props = iphase.props
            imol_oxide_comp = iphs_props['mol_oxide_comp']
            iendmember_name = iphs_props['endmember_name']
            iendmember_num = len(iendmember_name)
            ind_phs += 1

            iendmember_ind = np.arange(iendmember_num)

            iphase_name_tile = np.tile(np.array([iphs_props['phase_name']]), iendmember_num)

            all_phase_ind = np.hstack((all_phase_ind, np.tile(ind_phs,(iendmember_num))))
            all_mol_oxide_comp= np.vstack((all_mol_oxide_comp, imol_oxide_comp))
            all_phase_name = np.hstack((all_phase_name, iphase_name_tile))
            all_endmember_name = np.hstack((all_endmember_name, iendmember_name))
            all_endmember_ind = np.hstack((all_endmember_ind, iendmember_ind))

        return all_mol_oxide_comp

    def gibbs_energy_all(self, T, P, mols=None):
        mols = self._validate_mol_input(mols)

        gibbs_energy_all = []
        for iphs in self.phases:
            iphs_abbrev = iphs.abbrev
            imol = mols[iphs_abbrev]
            igibbs_energy = iphs.gibbs_energy(
                T, P, mol=imol)
            gibbs_energy_all.append(igibbs_energy)

        gibbs_energy_all = np.vstack(gibbs_energy_all)
        return gibbs_energy_all

    def chem_potential_all(self, T, P, mols=None):
        mols = self._validate_mol_input(mols)

        chem_potential_all = []
        for iphs in self.phases:
            iphs_abbrev = iphs.abbrev
            imol = mols[iphs_abbrev]
            ichem_potential = iphs.chem_potential(
                T, P, mol=imol)
            chem_potential_all.append(ichem_potential)


        chem_potential_all = np.vstack(chem_potential_all)
        return chem_potential_all

    def enthalpy_all(self, T, P):
        return np.vstack([phs.enthalpy(T, P)
                          for phs in self.phases])

    def entropy_all(self, T, P):
        return np.vstack([phs.entropy(T, P)
                          for phs in self.phases])

    def heat_capacity_all(self, T, P):
        return np.vstack([phs.heat_capacity(T, P)
                          for phs in self.phases])

    def dCp_dT_all(self, T, P):
        return np.vstack([phs.dCp_dT(T, P)
                          for phs in self.phases])

    def volume_all(self, T, P):
        return np.vstack([phs.volume(T, P)
                          for phs in self.phases])

    def dV_dT_all(self, T, P):
        return np.vstack([phs.dV_dT(T, P)
                          for phs in self.phases])

    def dV_dP_all(self, T, P):
        return np.vstack([phs.dV_dP(T, P)
                          for phs in self.phases])

    def d2V_dT2_all(self, T, P):
        return np.vstack([phs.d2V_dT2(T, P)
                          for phs in self.phases])

    def d2V_dTdP_all(self, T, P):
        return np.vstack([phs.d2V_dTdP(T, P)
                          for phs in self.phases])

    def d2V_dP2_all(self, T, P):
        return np.vstack([phs.d2V_dP2(T, P)
                          for phs in self.phases])
#===================================================
class classproperty(property):
    # https://stackoverflow.com/questions/128573/using-property-on-classmethods
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)
    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)
    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))

class Phase(metaclass=ABCMeta):
    """
    Abstract parent class defining generic phase properties.

    The user must use a subclass, like PurePhase or SolutionPhase, which
    implements the Phase interface.

    Parameters
    ----------
    phase_classnm : str
        Official class name for phase (implemented in src code). String has the
        form *classname* if source is *objc*, else:

          - if !calib ['cy', 'phase name', 'module name', '']

          - if  calib ['cy', 'phase name', 'module name', 'calib', '']
    abbrev : str
        Official abbreviation of phase (regardless of implementation).
    calib : bool, default True
        Indicates whether sample phase should be calibration ready.
    source : str
        Code source for phase implementation. Default is 'objc' (code in
        objective-C that is part of the original code base). Alternative is
        'coder' (code generated by the coder module).
    coder_module : str
        Name of the coder module that contains the phase classes. See
        documentation for model.Database for additional information and
        examples.

    Attributes
    ----------
    abbrev
    calib
    class_name
    endmember_ids
    endmember_names
    endmember_num
    formula
    identifier
    module
    MOLWTS
    OXIDES
    param_names
    param_props
    phase_name
    phase_obj
    phase_type
    props
    source

    Notes
    -----
    * This code is highly dependent on implementation and is likely to change
      dramatically with changes in the underlying code that calculates phase
      properties.
    * The pure water phase, "H2O", is very complex and thus not available for
      calibration. The water phase will force its calib flag to False, regardless
      of input value.

    """
    _MINVAL = np.sqrt(np.finfo(float).eps)

    def __init__(self, phase_classnm, abbrev, calib=True, source='objc',
        coder_module=None):
        if abbrev=='H2O':
            calib=False

        if source == 'coder':
            invalidate_caches()
            self._module = import_module(coder_module)
        else:
            self._module = None

        self._calib = calib
        self._source = source
        self.deriv_keys = {'dT','dP','dV','dS','dmol'}
        self._init_phase_props(phase_classnm, abbrev)
        self._init_param_props()
        self._phase_type = None
        self._abbrev = abbrev
        self._OXIDES = chem.oxide_props['oxides']
        self._MOLWTS = chem.oxide_props['molwt']


        # self._exchange_equil = equilibrate.ExchangeEquil(self)
        #Moved to class property
        #MINVAL = np.sqrt(np.finfo(float).eps)
        #self._MINVAL = MINVAL

        pass

    def _init_phase_props(self, phase_classnm, abbrev):

        if self.source == 'objc':
            phase_obj, phase_cls = core.get_src_object(
                phase_classnm, return_class=True)

            self._phase_obj = phase_obj
            self._phase_cls = phase_cls

            phase_name = str(phase_obj.phaseName)
            class_name = str(phase_cls.name)
            formula = str(phase_obj.phaseFormula)
            identifier = 'Objective-C-base'

            self._phase_name = phase_name
            self._class_name = class_name
            self._formula = formula
            self._identifier = identifier
        elif self.source == 'coder':
            # if !calib ['cy', 'phase name', 'module name']
            # if  calib ['cy', 'phase name', 'module name', 'calib']
            parts = phase_classnm.split("_")
            self._phase_obj = phase_classnm
            self._phase_cls = phase_classnm
            method = getattr(self.module, phase_classnm+'name')
            phase_name = method()
            self._phase_name = phase_name
            self._class_name = None
            try:
                method = getattr(self.module, phase_classnm+'formula')
                formula = method()
                self._formula = formula
            except AttributeError:
                self._formula = ""
                print ('Module generated by the coder package ' +
                'does not yet provide a formula method.')
            except TypeError:
                self._formula = ""
            method = getattr(self.module, phase_classnm+'identifier')
            identifier = method()
            self._identifier = identifier
            class_name = phase_classnm

        props = OrderedDict()
        props['abbrev'] = abbrev
        props['phase_name'] = phase_name
        props['class_name'] = class_name
        props['identifier'] = identifier
        props['endmember_name'] = [None]
        props['endmember_ids'] = [None] #TK
        props['formula'] = [None]
        props['atom_num'] = [None]
        props['molwt'] = [None]
        props['elemental_entropy'] = [None]
        props['element_comp'] = [None]
        props['mol_oxide_comp'] = [None]

        self._props = props
        pass

    def _init_param_props(self):
        if self.source == 'objc':
            try:
                phase_obj = self._phase_obj
                supports_calib = phase_obj.supportsParameterCalibration()
                param_num = phase_obj.getNumberOfFreeParameters()
                param_names_NSArray = phase_obj.getArrayOfNamesOfFreeParameters()
                param_names = [str(param_names_NSArray.objectAtIndex_(i))
                for i in range(param_num)]
                param_units = np.array([
                    str(phase_obj.getUnitsForParameterName_(key))
                    for key in param_names])
                param0 = np.array([
                    phase_obj.getValueForParameterName_(key)
                    for key in param_names])

            except AttributeError:
                supports_calib = False
                param_num = 0
                param_names = np.array([])
                param_units = np.array([])
                param0 = np.array([])
        else:
            if 'calib_' in self._phase_obj:
                supports_calib = True
                method_stub = self._phase_obj.replace('calib_', '')
                method = getattr(self.module, method_stub+'get_param_number')
                param_num = method()
                method = getattr(self.module, method_stub+'get_param_names')
                param_names = method()
                method = getattr(self.module, method_stub+'get_param_units')
                param_units = method()
                method = getattr(self.module, method_stub+'get_param_values')
                param0 = method()
            else:
                supports_calib = False
                param_num = 0
                param_names = np.array([])
                param_units = np.array([])
                param0 = np.array([])

        param_props = OrderedDict()
        param_props['supports_calib'] = supports_calib
        param_props['param_num'] = param_num
        param_props['param_names'] =  param_names
        param_props['param_units'] =  param_units
        param_props['param0'] = param0

        self._param_props = param_props
        pass

    def __eq__(self, other):
        return self._props['abbrev'] == other._props['abbrev']

    def __lt__(self, other):
        return self._props['abbrev'] <= other._props['abbrev']

    def __gt__(self, other):
        return self._props['abbrev'] >= other._props['abbrev']

    @property
    def exchange_equil(self):
        """
        Exchange equilibrium object responsible for calculating
        metastable equilibrium properties of the phase

        Returns
        -------
        Exchange Equilibrium object

        """

        return self._exchange_equil

    @property
    def phase_name(self):
        """
        Name of phase

        Returns
        -------
        Name of phase (str)

        """

        return self._phase_name

    @property
    def abbrev(self):
        """
        Official unique abbreviation for phase

        Returns
        -------
        Abbreviation (str)

        """

        return self._abbrev

    @property
    def class_name(self):
        """
        Name of class

        Returns
        -------
        Name of class (str)

        """

        return self._class_name

    @property
    def formula(self):
        """
        Formula of phase

        Returns
        -------
        Formula of phase (str)

        """

        return self._formula

    @property
    def identifier(self):
        """
        Identifier of phase

        Returns
        -------
        Identifier of phase (str)

        """

        return self._identifier

    @property
    def phase_type(self):
        """
        Phase type

        Returns
        -------
        Phase type (str)
            Permissible values are 'pure' or 'solution'.

        """

        return self._phase_type

    @property
    def calib(self):
        """
        Indicates whether phase calibration is enabled

		Returns
        -------
        Value of calib (bool)

        """

        return self._calib

    @property
    def source(self):
        """
        Indicates origin of source code implementation

        Returns
        -------
        String indicating origin of source code for implementation
            Permissible values 'objc' or 'coder'.

        """

        return self._source

    @property
    def module(self):
        """
        Python module attribute for coder generated functions

        Returns
        -------
        module
            Module attribute returned from importlib.import_module,
            else None if source is 'objc'

        """

        return self._module

    @property
    def props(self):
        """
        Dictionary of phase properties

        The dictionary defines phase properties with these keys:

        ``abbrev`` : str
            Official unique phase abbreviation
        ``name`` : str
            Name of phase (implementation dependent)
        ``class_name`` : str
            Official class name for phase (implemented in src code)
        ``formula`` : str
            Formula of phase
        ``natom`` : int
            Number of atoms in formula unit
        ``molwt`` : double
            Molecular weight of phase (in g/mol-formula-unit)
        ``elemental_entropy`` : double
            Estimated entropy from elemental formula (from Robie et al. 1979)
        ``element_symbols`` : str array
            Symbol array string
        ``element_comp`` : int array
            Phase formula in terms of number of each element

		Returns
        -------
        A Python dictionary (dict)

        Notes
        -----
        Need to update these dictionary values to be vectors for solution phases

        """

        return self._props

    @property
    def endmember_names(self):
        """
        Name of each endmember

        Returns
        -------
        List of endmember names for this solution phase, [str,...]

        """

        return self.props['endmember_name']

    @property
    def endmember_num(self):
        """
        Number of endmembers in phase

        Returns
        -------
        Number of endmembers in phase (int)

        """

        return len(self.props['endmember_name'])

    @property
    def endmember_ids(self):
        """
        ID number of each endmember in phase

        Returns
        -------
        Array of ids, [int,...]

        """

        return self.props['endmember_ids'] #TK

    @property
    def param_props(self):
        """
        Dictionary of phase model parameters

        This dictionary defines parameter properties for the phase, using these keys:

        ``supports_calib`` : bool
            Flag indicating whether phase allows calibration
        ``param_num`` : int
            Number of parameters
        ``param_names`` : str array
            Name of each parameter
        ``param_units`` : str array
            Units for each parameter
        ``param0`` : double array
            Initial parameter values

 	    Returns
        -------
        Dictionary of phase model parameters (dict)

        """

        return self._param_props

    @property
    def phase_obj(self):
        """
        Instance of the phase object

        Returns
        -------
        Object instance

        """

        return self._phase_obj

    @property
    def param_names(self):
        """
        Array of parameter names

        Returns
        -------
        Array of names for each parameter of the phase model, [str,...]

        """

        return self.param_props['param_names']

    def param_units(self, param_names=[],
                    all_params =False):
        """
        Get units for listed parameters.

        Parameters
        ----------
        param_names : str array
            List of parameter names
        all_params : bool, default False
            If true, returns units for all parameters

        Returns
        -------
        units : double array
            List of units for selected parameters

        """

        if all_params:
            return self.param_props['param_units']

        if type(param_names) == str:
            param_names = [param_names]

        units_l = []
        for ind,key in enumerate(param_names):
            unit = self.param_props['param_units'][ind]
            # unit = self._phase_obj.getUnitsForParameterName_(key)
            units_l.append(unit)

        return units_l

    @property
    def OXIDES(self):
        """
        Array of oxide names

  		Returns
        -------
        Array of oxide names, [str,...]

        """

        return self._OXIDES

    @property
    def MOLWTS(self):
        """
        Array of molecular weights of oxides

        Returns
        -------
        Numpy array of molecular weights, (nparray)

        """

        return self._MOLWTS

    @classproperty
    def MINVAL(cls):
        """
        Minimum molar concentration of a component in a solution phase

        Returns
        -------
        Floating point number
        """
        return cls._MINVAL

    @MINVAL.setter
    def MINVAL(cls, value):
        assert value > np.finfo(float).eps, 'MINVAL must have a value ' \
            + 'greater than machine precision(' + str(np.finfo(float).eps) + ')'
        cls._MINVAL = value

    def set_ref_state(self, Tr=298.15, Pr=1.0, Trl=298.15):
        """
        Set reference state P/T conditions.

        Parameters
        ----------
        Tr : double, default 298.15
            Reference temperature in Kelvin
        Pr : double, default 1.0
            Reference pressure in bars
        Trl : double, default 298.15
            Reference temperature for lambda heat capacity correction in Kelvin

        """
        if self.source == 'objc':
            self._phase_obj.setTr_(Tr)
            self._phase_obj.setPr_(Pr)
            self._phase_obj.setTrl_(Trl)
        pass

    def enable_gibbs_energy_reference_state(self):
        """
        Set Gibbs energy of the reference state.

        Notes
        -----
        Call method on any phase class, and it automatically applies to all.

        """
        if self.source == 'objc':
            phase_cls = self._phase_cls
            phase_cls.enableGibbsFreeEnergyReferenceStateUsed()
        pass

    def disable_gibbs_energy_reference_state(self):
        """
        Unset Gibbs energy of the reference state.

        Notes
        -----
        Call method on any phase class, and it automatically applies to all.

        """
        # call method on any phase class (automatically applied to all)
        if self.source == 'objc':
            phase_cls = self._phase_cls
            phase_cls.disableGibbsFreeEnergyReferenceStateUsed()
        pass

    def get_param_values(self, param_names=[], all_params=False):
        """
        Get current values for listed parameters.

        Parameters
        ----------
        param_names : str array
            List of parameter names
        all_params : bool, default False
            If true, returns units for all parameters

        Returns
        -------
        values : double array
            List of values for selected parameters

        """

        if all_params:
            param_names = self.param_names
        elif type(param_names) == str:
            param_names = [param_names]

        values_a = np.zeros(len(param_names))

        if self.source == 'objc':
            for ind,key in enumerate(param_names):
                value = self._phase_obj.getValueForParameterName_(key)
                values_a[ind] = value
        elif self.calib:
            method_stub = self._phase_obj.replace('calib_', '')
            method = getattr(self.module, method_stub+'get_param_value')
            for ind,key in enumerate(param_names):
                value = method(ind)
                values_a[ind] = value
        else:
            values_a = None

        return values_a

    def set_param_values(self, param_names=[], param_values=[]):
        """
        Set new values for listed parameters.

        Parameters
        ----------
        param_names : str array
            List of parameter names
        param_values : double array
            List of parameter values

        """
        assert len(param_names)==len(param_values), \
            'param_names and param_values must have the same length'

        if self.source == 'objc':
            for name, value in zip(param_names, param_values):
                self._phase_obj.setParameterName_tovalue_(name, float(value))
        elif self.calib:
            method_stub = self._phase_obj.replace('calib_', '')
            method = getattr(self.module, method_stub+'set_param_value')
            for name, value in zip(param_names, param_values):
                method(name, value)

        pass

    def get_phase_amount_from_elements(self, elements, kind='mass'):
        """
        Convert list of elements to quantity of phase.

        Parameters
        ----------
        elements : double array
            Number of each element
        kind : {'mass','moles'}
            Determines how phase amount is determined (mass vs. moles)

        Returns
        -------
        amount : double
            Amount of phase (expressed according to kind)

        """

        if self.source == 'objc':
            elements_vec = core.array_to_double_vector(elements)

            # NOTE: Need to update to allow for solution phases
            if kind == 'mass':
                return self._phase_obj.convertElementsToMassOfPhase_(
                    elements_vec)
            elif kind == 'moles':
                return self._phase_obj.convertElementsToMolesOfPhase_(
                    elements_vec)
            else :
                raise NotImplemented('The kind "'+kind+'" is not implimented ' +
                    'for get_phase_amount_from_elements()')
        else:
            raise NotImplemented('Function not implemented for coder ' +
                'generated modules.')

    def affinity_and_comp(self, t, p, mu, **kwargs):

        exch_equil = self.exchange_equil
        return exch_equil.affinity_and_comp(t, p, mu, **kwargs)

    def affinity_and_comp_legacy(self, t, p, mu, **kwargs):
        exch_equil = self.exchange_equil
        return exch_equil.affinity_and_comp_legacy(t, p, mu, **kwargs)


    def determine_phase_stability(self, t, p, moles, **kwargs):

        exch_equil = self.exchange_equil
        return exch_equil.determine_phase_stability(t, p, moles, **kwargs)

    def _validate_deriv(self, deriv):
        if not deriv:
            deriv = {}
        else:
            assert all([key in self.deriv_keys for key in deriv]), \
                'Specified deriv keys, '+str(deriv)+' are not allowed. ' + \
                'They must belong to the allowable set: '+str(self.deriv_keys)

        # Check for all zeros (default)
        if all(deriv[key]==0 for key in deriv):
            deriv = {}

        '''
        Insert some tests here for values of derivative keys:
        'dT' = <3  'dP' = <3 'dmol' = <2 'dw' <= 1
        '''

        return deriv

    def _nudge_solution_comp(self, mol):
        """
        Adjust solution composition to ensure that all components are present
        (at machine precision).

        """

        MINVAL = self.MINVAL
        ind_row, ind_col = np.where(mol==0)
        for irow, icol in zip(ind_row, ind_col):
            mol[ind_row, ind_col] = MINVAL

        return mol

    ####################################################
    # Decorator for validating thermodynamic functions #
    ####################################################
    def validate(func):
        # *args arguments as tuple for a in args
        # **kwargs dictionary of keyword arguments for a in kwargs
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # execute before function call
            self = args[0]
            T             = args[1]
            P             = args[2]
            mol           = kwargs['mol'] if ('mol' in kwargs) else None
            V             = kwargs['V'] if ('V' in kwargs) else None
            deriv         = kwargs['deriv'] if ('deriv' in kwargs) else None
            deriv_param   = kwargs['deriv_param'] if (
                'deriv_param' in kwargs) else None
            mol_deriv_qty = kwargs['mol_deriv_qty'] if (
                'mol_deriv_qty' in kwargs) else False
            endmember     = kwargs['endmember'] if (
                'endmember' in kwargs) else None
            const = kwargs['const'] if 'const' in kwargs else None
            species = kwargs['species'] if 'species' in kwargs else None
            if self.phase_type == 'pure':
                assert not mol, 'mol must be empty for PurePhase.'

            deriv = {'dT':0, 'dP':0, 'dV':0, 'dS':0, 'dmol':0} if (
                deriv is None) else deriv
            endmember = -1 if endmember is None else endmember

            if deriv_param is not None:
                if not self._param_props['supports_calib']:
                    assert("This module does not support calibration.")
                if type(deriv_param) is int:
                    deriv_param = [deriv_param]
                elif type(deriv_param) is str:
                    if deriv_param in self._param_props['param_names']:
                        deriv_param = (
                            self._param_props['param_names'].index(deriv_param))
                    else:
                        assert("Parameter name is in param_props list. ")
                    deriv_param = [deriv_param]
                elif type(deriv_param) is list:
                    for ind,(x) in enumerate(deriv_param):
                        if type(x) is str:
                            if x in self._param_props['param_names']:
                                deriv_param[ind] = (
                                    self._param_props['param_names'].index(x))
                            else:
                                assert("Parameter name is in param_props list. ")
                elif type(deriv_param) is np.ndarray:
                    pass
                else:
                    print ('deriv_param argument must be one of int, ' +
                        'str, list, or numpy array.')

            scalar_input = False
            if V is None:
                T, P = core.fill_array(T, P)
                if len(T)==1 & len(P)==1:
                    scalar_input = True

            else:   # overide pressure with input volume
                P = None
                T, V = core.fill_array(T, V)
                if len(T)==1 & len(V)==1:
                    scalar_input = True

            if mol_deriv_qty and endmember is not None:
                mol_deriv_qty = False

            T, endmember = core.fill_array(T, endmember)

            deriv = self._validate_deriv(deriv)

            if 'dmol' in deriv and deriv['dmol']>0:
                scalar_input = False

            if mol is not None:
                mol = np.array(mol)
                num_PT_pts = len(T)
                if mol.ndim==1:
                    mol = np.tile(mol[np.newaxis, :], (num_PT_pts, 1))
                elif mol.ndim==2:
                    assert mol.shape[0]==num_PT_pts, ('molar composition must '
                        + 'be provided for every PT point, or otherwise be '
                        + 'fixed to a single value for all PT points.')
                else:
                    assert False, ('Molar composition must be defined, for '
                        + 'each PT point. It currently has too many '
                        + 'dimensions.')
                mol = self._nudge_solution_comp(mol)
                assert mol.size > 0, ('mol cannot be initialized to an empty '
                    + 'numpy array or to an empty list')

            # call the function that is wrapped
            if const:
                result = func(self, T, P, mol=mol, V=V, deriv=deriv,
                    deriv_param=deriv_param, mol_deriv_qty=mol_deriv_qty,
                    endmember=endmember, const=const)
            elif species:
                result = func(self, T, P, mol=mol, V=V, deriv=deriv,
                    deriv_param=deriv_param, mol_deriv_qty=mol_deriv_qty,
                    endmember=endmember, species=species)
            else:
                result = func(self, T, P, mol=mol, V=V, deriv=deriv,
                    deriv_param=deriv_param, mol_deriv_qty=mol_deriv_qty,
                    endmember=endmember)

            # execute after function call
            if result is None:
                raise NotImplementedError('Cannot find specified derivative: '
                                 +str(deriv))
            if scalar_input and not mol_deriv_qty:
                result = result[0]

            return result
        return func_wrapper

    ###################
    # State Functions #
    ###################

    ###########
    # order 0 #
    ###########
    @validate
    def gibbs_energy(self, T, P, mol=None, V=None, deriv=None,
                     deriv_param=None, mol_deriv_qty=None, endmember=None,
                     const=None, species=None):
        """
        Calculate Gibbs energy (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin
        P : array-like
            Pressure in bars
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases)
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (Default is zero for all params.)
        deriv_param : array of strs
            Parameter names that identify returned derivatives
            (Default is None.)

        Returns
        -------
        value : array-like
            Gibbs energy in J (or derivative in deriv units)

        """

        method  = None
        args_dw = False
        if (not deriv) and (not deriv_param):
            method = getattr(self, '_calc_G')
        else:
            nT   = deriv['dT'] if ('dT' in deriv) else 0
            nP   = deriv['dP'] if ('dP' in deriv) else 0
            nMol = deriv['dmol'] if ('dmol' in deriv) else 0
            nW   = 0 if not deriv_param else 1
            nTot = nT + nP + nMol + nW
            method_sig = '_calc_'
            method_sig += 'd' + str(nTot) + 'G_' if (nTot > 1) else (
                'dG_' if (nTot > 0) else 'G')
            method_sig += 'dT' + str(nT) if (nT > 1) else (
                'dT' if (nT > 0) else '')
            method_sig += 'dP' + str(nP) if (nP > 1) else (
                'dP' if (nP > 0) else '')
            method_sig += 'dm' + str(nMol) if (nMol > 1) else (
                'dm' if (nMol > 0) else '')
            method_sig += 'dw' if (nW > 0) else ''
            method = getattr(self, method_sig)
            args_dw = False if nW == 0 else True

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, param=deriv_param
                        ) if args_dw else method(iT, iP, V=V))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, param=deriv_param
                        ) if args_dw else method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def enthalpy(self, T, P, mol=None, V=None, deriv=None,
                     deriv_param=None, mol_deriv_qty=None, endmember=None,
                     const=None, species=None):
        """
        Calculate enthalpy (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin
        P : array-like
            Pressure in bars
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Enthalpy in J (or derivative in deriv units).

        """

        method  = None
        # Fix me!
        if (not deriv) and (not deriv_param):
            method = getattr(self, '_calc_H')
        else:
            nT   = deriv['dT'] if ('dT' in deriv) else 0
            nP   = deriv['dP'] if ('dP' in deriv) else 0
            nMol = deriv['dmol'] if ('dmol' in deriv) else 0
            nW   = 0 if not deriv_param else 1
            nTot = nT + nP + nMol + nW
            assert nT == 0 and nP == 0 and nMol == 0 and nW == 1, \
                'Only parameter derivatives of the enthalpy are permitted'
            method_sig = '_calc_'
            method_sig += 'd' + str(nTot) + 'H_' if (nTot > 1) else (
                'dH_' if (nTot > 0) else 'H')
            method_sig += 'dw' if (nW > 0) else ''
            method = getattr(self, method_sig)

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V))
            else:
                for ind,(iT,iP, imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def helmholtz_energy(self, T, P, mol=None, V=None, deriv=None,
                         deriv_param=None, mol_deriv_qty=None, endmember=None,
                         const=None, species=None):
        """
        Calculate helmholtz energy (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Helmholtz energy in J (or derivative in deriv units).

        """
        return None

    @validate
    def internal_energy(self, T, P, mol=None, V=None, deriv=None,
        deriv_param=None, mol_deriv_qty=None, endmember=None,
        const=None, species=None):
        """
        Calculate internal energy (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Internal energy in J (or derivative in deriv units).

        """
        return None

    ###########
    # order 1 #
    ###########
    @validate
    def volume(self, T, P, mol=None, V=None, deriv=None,
               deriv_param=None, mol_deriv_qty=None, endmember=None,
               const=None, species=None):
        """
        Calculate volume (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Volume in J/bar (or derivative in deriv units).

        """

        method  = None
        #fix me
        if (not deriv) and (not deriv_param):
            method = getattr(self, '_calc_V')
        else:
            nT   = deriv['dT'] if ('dT' in deriv) else 0
            nP   = deriv['dP'] if ('dP' in deriv) else 0
            nMol = deriv['dmol'] if ('dmol' in deriv) else 0
            nW   = 0 if not deriv_param else 1
            nTot = nT + nP + nMol + nW
            method_sig = '_calc_'
            method_sig += 'd' + str(nTot) + 'V_' if (nTot > 1) else (
                'dV_' if (nTot > 0) else 'V')
            method_sig += 'dm' + str(nMol) if (nMol > 1) else (
                'dm' if (nMol > 0) else '')
            method_sig += 'dT' + str(nT) if (nT > 1) else (
                'dT' if (nT > 0) else '')
            method_sig += 'dP' + str(nP) if (nP > 1) else (
                'dP' if (nP > 0) else '')
            method_sig += 'dw' if (nW > 0) else ''
            method = getattr(self, method_sig)

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def density(self, T, P, mol=None, V=None, deriv=None,
                deriv_param=None, mol_deriv_qty=None, endmember=None,
                const=None, species=None):
        """
        Calculate density (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Density in g*bar/J (or derivative in deriv units).

        """

        method  = None
        if (not deriv) and (not deriv_param):
            method = getattr(self, '_calc_density')

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def entropy(self, T, P, mol=None, V=None, deriv=None,
                deriv_param=None, mol_deriv_qty=None, endmember=None,
                const=None, species=None):
        """
        Calculate entropy (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Entropy in J/K (or derivative in deriv units).

        """

        method = None
        #fix me
        if (not deriv) and (not deriv_param):
            method = getattr(self, '_calc_S')
        else:
            nT   = deriv['dT'] if ('dT' in deriv) else 0
            nP   = deriv['dP'] if ('dP' in deriv) else 0
            nMol = deriv['dmol'] if ('dmol' in deriv) else 0
            nW   = 0 if not deriv_param else 1
            nTot = nT + nP + nMol + nW
            method_sig = '_calc_'
            method_sig += 'd' + str(nTot) + 'S_' if (nTot > 1) else (
                'dS_' if (nTot > 0) else 'S')
            method_sig += 'dm' + str(nMol) if (nMol > 1) else (
                'dm' if (nMol > 0) else '')
            method_sig += 'dT' + str(nT) if (nT > 1) else (
                'dT' if (nT > 0) else '')
            method_sig += 'dP' + str(nP) if (nP > 1) else (
                'dP' if (nP > 0) else '')
            method_sig += 'dw' if (nW > 0) else ''
            method = getattr(self, method_sig)

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def chem_potential(self, T, P, mol=None, V=None, deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None, species=False,
        const=None):
        """
        Calculate chemical potential (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).
        endmember : None or int scalar or int array
            If None, retrieve an array of chemical potentials, else
            chemical potential for endmember index or index set in array
        species : boolean
            If False, returned value is for components of the solution.
            If True, returned value is for species in the solution.

        Returns
        -------
        value : array-like
            Chemical potential in J (or derivative in deriv units).

        """
        if not deriv:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(self._calc_mu(iT, iP, V=V,
                        endmember=endmember, species=species))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(self._calc_mu(iT, iP, mol=imol, V=V,
                        endmember=endmember, species=species))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def activity(self, T, P, mol=None, V=None, deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None,
        const=None, species=None):
        """
        Calculate activity (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).
        endmember : None or int scalar or int array
            If None, retrieve an array of chemical potentials; else
            chemical potential for endmber index or index set in array

        Returns
        -------
        value : array-like
            Activity (or derivatives in deriv units).

        """

        method = None
        if not deriv:
            method = getattr(self, '_calc_a')
        elif deriv=={'dmol':1}:
            method = getattr(self, '_calc_da_dm')
        else:
            pass

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V, endmember=endmember))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V,
                        endmember=endmember))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def fugacity(self, T, P, mol=None, V=None, deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None,
        const=None, species=None):
        """
        Calculate activity (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Fugacity in bars (or derivatives in deriv units).

        """
        return None

    ###########
    # order 2 #
    ###########
    @validate
    def thermal_exp(self, T, P, mol=None, V=None, deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None,
        const=None, species=None):
        """
        Calculate heat capacity (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Thermal expansion in 1/K (or derivatives in deriv units).

        """

        method  = None
        if (not deriv) and (not deriv_param):
            method = getattr(self, '_calc_alpha')

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def gamma(self, T, P, mol=None, V=None, deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None,
        const=None, species=None):
        """
        Calculate grüneisen parameter (or derivatives) for phase.

        NOT IMPLEMENTED

        """
        return None

    @validate
    def heat_capacity(self, T, P, mol=None, V=None, const='P', deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None, species=None):
        """
        Calculate heat capacity (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        const : ['P', 'V'], optional
            Defines constant path for derivative (yielding C_P vs C_V)
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Thermal expansion in J/K (or derivatives in deriv units).

        """
        method = None
        if const=='P':
            if not deriv:
                method = getattr(self, '_calc_Cp')
            elif deriv=={'dT':1}:
                method = getattr(self, '_calc_dCp_dT')
            elif deriv=={'dmol':1}:
                method = getattr(self, '_calc_dCp_dm')
            else:
                result = None
        elif const=='V':
            if not deriv:
                method = getattr(self, '_calc_Cv')

        if method:
            result = []
            if mol is None:
                for ind,(iT,iP) in enumerate(zip(T,P)):
                    result.append(method(iT, iP, V=V))
            else:
                for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                    result.append(method(iT, iP, mol=imol, V=V))
            result = np.array(result)
        else:
            result = None

        return result

    @validate
    def bulk_mod(self, T, P, mol=None, V=None, const='T', deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None, species=None):
        """
        Calculate bulk modulus (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        const : ['T', 'S'], optional
            Defines constant path for derivative (yielding K_T vs K_S)
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Bulk modulus in bars (or derivatives in deriv units).

        """
        if const=='T':
            method = None
            if (not deriv) and (not deriv_param):
                method = getattr(self, '_calc_K')
            else:
                nP   = deriv['dP'] if ('dP' in deriv) else 0
                if nP == 1:
                    method = getattr(self, '_calc_Kp')

            if method:
                result = []
                if mol is None:
                    for ind,(iT,iP) in enumerate(zip(T,P)):
                        result.append(method(iT, iP, V=V))
                else:
                    for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                        result.append(method(iT, iP, mol=imol, V=V))
                result = np.array(result)
            else:
                result = None
        else: # const=='S'
            result = None

        return result

    @validate
    def compressibility(self, T, P, mol=None, V=None, const='T', deriv=None,
        deriv_param=None, mol_deriv_qty=True, endmember=None, species=None):
        """
        Calculate compressibility (or derivatives) for phase.

        Parameters
        ----------
        T : array-like
            Temperature in Kelvin.
        P : array-like
            Pressure in bars.
        mol : array-like, optional
            Composition in terms of mols of endmembers
            (unneeded for pure phases).
        const : ['T', 'S'], optional
            Defines constant path for derivative (yielding Beta_T vs Beta_S)
        V : array-like, optional (default None)
            Volume in J/bar. Overrides pressure if not None.
        deriv : dict of ints
            Derivative order for each parameter
            (default is zero for all params).

        Returns
        -------
        value : array-like
            Bulk modulus in 1/bars (or derivatives in deriv units).

        """
        if const=='T':
            method = None
            if (not deriv) and (not deriv_param):
                method = getattr(self, '_calc_beta')

            if method:
                result = []
                if mol is None:
                    for ind,(iT,iP) in enumerate(zip(T,P)):
                        result.append(method(iT, iP, V=V))
                else:
                    for ind,(iT,iP,imol) in enumerate(zip(T,P,mol)):
                        result.append(method(iT, iP, mol=imol, V=V))
                result = np.array(result)
            else:
                result = None
        else: # const=='S'
            result = None

        return result

    ######################
    # Calculator methods #
    ######################
    @abstractmethod
    def _calc_G(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dG_dT(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dG_dP(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dG_dm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dG_dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d2G_dT2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2G_dTdP(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2G_dTdm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2G_dTdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d2G_dP2(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d2G_dPdm(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d2G_dPdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d2G_dm2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2G_dmdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dT3(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dT2dP(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dT2dm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dT2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dTdP2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dTdPdm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dTdPdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dTdm2(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dTdmdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dP3(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dP2dm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dP2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dPdm2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dPdmdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d3G_dm3(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d3G_dm2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dT2dmdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dTdPdmdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dP2dmdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dm3dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dTdm2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dPdm2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dT3dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dT2dPdw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dTdP2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d4G_dP3dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d5G_dT2dm2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d5G_dTdPdm2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_d5G_dP2dm2dw(self, T, P, mol=None, param=None):
        pass

    @abstractmethod
    def _calc_H(self, T, P, mol=None, V=None):
        pass

    # (order=1)
    @abstractmethod
    def _calc_S(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_V(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_Cv(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_Cp(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dCp_dT(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dV_dT(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dV_dP(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2V_dT2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2V_dTdP(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2V_dP2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_density(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_alpha(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_beta(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_K(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_Kp(self, T, P, mol=None, V=None):
        pass

    # Compositional quantities
    @abstractmethod
    def _calc_mu(self, T, P, mol=None, V=None, endmember=None, species=False):
        pass

    @abstractmethod
    def _calc_a(self, T, P, mol=None, V=None, endmember=None):
        pass

    @abstractmethod
    def _calc_dV_dm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_dS_dm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_da_dm(self, T, P, mol=None, V=None, endmember=None):
        pass

    # (order=3)
    @abstractmethod
    def _calc_dCp_dm(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2V_dmdT(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2V_dmdP(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2S_dm2(self, T, P, mol=None, V=None):
        pass

    @abstractmethod
    def _calc_d2V_dm2(self, T, P, mol=None, V=None):
        pass

#===================================================
class PurePhase(Phase):
    """
    Pure stoichiometric phases.

    Implements the Phase interface.

    Parameters
    ----------
    phase_classnm : str
        Official class name for phase (implemented in src code)
    abbrev : str
        Official abbreviation of phase (regardless of implementation)
    calib : bool, default True
        Indicates whether sample phase should be calibration ready

    Attributes
    ----------
    Berman_formula

    Notes
    -----
    * This code is highly dependent on implementation and is likely to change
      dramatically with changes in the underlying code that calculates phase
      properties.
    * The pure water phase, "H2O", is very complex and thus not available for
      calibration. The water phase will force its calib flag to False, regardless
      of input value.
    * In addition to the attributes listed, this class inherits the Phase class
      attributes.

    """

    def __init__(self, phase_classnm, abbrev, calib=True, source='objc',
        coder_module=None):
        super().__init__(phase_classnm, abbrev, calib=calib, source=source,
            coder_module=coder_module)
        self._init_pure_phase_props(phase_classnm)
        self._phase_type = 'pure'

        self._methods = {}
        if self.source == 'objc':
            self._methods['_calc_G'] = getattr(self._phase_obj,
                'getGibbsFreeEnergyFromT_andP_')
            self._methods['_calc_dG_dT'] = getattr(self, 'not_coded')
            self._methods['_calc_dG_dP'] = getattr(self._phase_obj,
                'getVolumeFromT_andP_')
            self._methods['_calc_dG_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_dG_dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d2G_dT2'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dTdP'] = getattr(self._phase_obj,
                'getDvDtFromT_andP_')
            self._methods['_calc_d2G_dTdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dTdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dP2'] = getattr(self._phase_obj,
                'getDvDpFromT_andP_')
            self._methods['_calc_d2G_dPdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dPdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dmdw'] = getattr(self, 'not_coded')

            self._methods['_calc_d3G_dT3'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dT2dP'] = getattr(self._phase_obj,
                'getD2vDt2FromT_andP_')
            self._methods['_calc_d3G_dT2dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dT2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdP2'] = getattr(self._phase_obj,
                'getD2vDtDpFromT_andP_')
            self._methods['_calc_d3G_dTdPdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdPdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dP3'] = getattr(self._phase_obj,
                'getD2vDp2FromT_andP_')
            self._methods['_calc_d3G_dP2dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dP2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dPdm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dm3'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dP2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dm3dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dPdm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT3dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dT2dPdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdP2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dP3dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d5G_dT2dm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dTdPdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dP2dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_H'] = getattr(self._phase_obj,
                'getEnthalpyFromT_andP_')
            self._methods['_calc_S'] = getattr(self._phase_obj,
                'getEntropyFromT_andP_')
            self._methods['_calc_dS_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2S_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_V'] = getattr(self._phase_obj,
                'getVolumeFromT_andP_')
            self._methods['_calc_dV_dT'] = getattr(self._phase_obj,
                'getDvDtFromT_andP_')
            self._methods['_calc_dV_dP'] = getattr(self._phase_obj,
                'getDvDpFromT_andP_')
            self._methods['_calc_dV_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2V_dT2'] = getattr(self._phase_obj,
                'getD2vDt2FromT_andP_')
            self._methods['_calc_d2V_dTdP'] = getattr(self._phase_obj,
                'getD2vDtDpFromT_andP_')
            self._methods['_calc_d2V_dP2'] = getattr(self._phase_obj,
                'getD2vDp2FromT_andP_')
            self._methods['_calc_d2V_dmdT'] = getattr(self, 'not_coded')
            self._methods['_calc_d2V_dmdP'] = getattr(self, 'not_coded')
            self._methods['_calc_d2V_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_Cv'] = getattr(self, 'not_coded')
            self._methods['_calc_Cp'] = getattr(self._phase_obj,
                'getHeatCapacityFromT_andP_')
            self._methods['_calc_dCp_dT'] = getattr(self._phase_obj,
                'getDcpDtFromT_andP_')
            self._methods['_calc_dCp_dm'] = getattr(self, 'not_coded')

            self._methods['_calc_density'] = getattr(self, 'not_coded')
            self._methods['_calc_alpha'] = getattr(self, 'not_coded')
            self._methods['_calc_beta'] = getattr(self, 'not_coded')
            self._methods['_calc_K'] = getattr(self, 'not_coded')
            self._methods['_calc_Kp'] = getattr(self, 'not_coded')

            self._methods['_calc_mu'] = getattr(self._phase_obj,
                'getGibbsFreeEnergyFromT_andP_')
            self._methods['_calc_a'] = getattr(self, 'not_coded')
            self._methods['_calc_da_dm'] = getattr(self, 'not_coded')

        else:
            phase_calibnm = phase_classnm.replace('calib', 'dparam') if (
                self.calib) else ''
            self._methods['_calc_G'] = getattr(self.module,
                phase_classnm+'g')
            self._methods['_calc_dG_dT'] = getattr(self.module,
                phase_classnm+'dgdt')
            self._methods['_calc_dG_dP'] = getattr(self.module,
                phase_classnm+'dgdp')
            self._methods['_calc_dG_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_dG_dw'] = getattr(self.module,
                phase_calibnm+'g') if self.calib else getattr(
                self, 'not_coded')

            self._methods['_calc_d2G_dT2'] = getattr(self.module,
                phase_classnm+'d2gdt2')
            self._methods['_calc_d2G_dTdP'] = getattr(self.module,
                phase_classnm+'d2gdtdp')
            self._methods['_calc_d2G_dTdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dTdw'] = getattr(self.module,
                phase_calibnm+'dgdt') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d2G_dP2'] = getattr(self.module,
                phase_classnm+'d2gdp2')
            self._methods['_calc_d2G_dPdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dPdw'] = getattr(self.module,
                phase_calibnm+'dgdp') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d2G_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dmdw'] = getattr(self, 'not_coded')

            self._methods['_calc_d3G_dT3'] = getattr(self.module,
                phase_classnm+'d3gdt3')
            self._methods['_calc_d3G_dT2dP'] = getattr(self.module,
                phase_classnm+'d3gdt2dp')
            self._methods['_calc_d3G_dT2dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dT2dw'] = getattr(self.module,
                phase_calibnm+'d2gdt2') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d3G_dTdP2'] = getattr(self.module,
                phase_classnm+'d3gdtdp2')
            self._methods['_calc_d3G_dTdPdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdPdw'] = getattr(self.module,
                phase_calibnm+'d2gdtdp') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d3G_dTdm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dP3'] = getattr(self.module,
                phase_classnm+'d3gdp3')
            self._methods['_calc_d3G_dP2dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dP2dw'] = getattr(self.module,
                phase_calibnm+'d2gdp2') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d3G_dPdm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dm3'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dP2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dm3dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dPdm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT3dw'] = getattr(self.module,
                phase_calibnm+'d3gdt3') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dT2dPdw'] = getattr(self.module,
                phase_calibnm+'d3gdt2dp') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dTdP2dw'] = getattr(self.module,
                phase_calibnm+'d3gdtdp2') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dP3dw'] = getattr(self.module,
                phase_calibnm+'d3gdp3') if self.calib else getattr(
                self, 'not_coded')

            self._methods['_calc_d5G_dT2dm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dTdPdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dP2dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_H'] = getattr(self, 'not_coded')
            self._methods['_calc_S'] = getattr(self.module,
                phase_classnm+'s')
            self._methods['_calc_dS_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2S_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_V'] = getattr(self.module,
                phase_classnm+'v')
            self._methods['_calc_dV_dT'] = getattr(self.module,
                phase_classnm+'d2gdtdp')
            self._methods['_calc_dV_dP'] = getattr(self.module,
                phase_classnm+'d2gdp2')
            self._methods['_calc_dV_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2V_dT2'] = getattr(self.module,
                phase_classnm+'d3gdt2dp')
            self._methods['_calc_d2V_dTdP'] = getattr(self.module,
                phase_classnm+'d3gdtdp2')
            self._methods['_calc_d2V_dP2'] = getattr(self.module,
                phase_classnm+'d3gdp3')
            self._methods['_calc_d2V_dmdT'] = getattr(self, 'not_coded')
            self._methods['_calc_d2V_dmdP'] = getattr(self, 'not_coded')
            self._methods['_calc_d2V_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_Cv'] = getattr(self.module,
                phase_classnm+'cv')
            self._methods['_calc_Cp'] = getattr(self.module,
                phase_classnm+'cp')
            self._methods['_calc_dCp_dT'] = getattr(self.module,
                phase_classnm+'dcpdt')
            self._methods['_calc_dCp_dm'] = getattr(self, 'not_coded')

            self._methods['_calc_density'] = getattr(self, 'not_coded')
            self._methods['_calc_alpha'] = getattr(self.module,
                phase_classnm+'alpha')
            self._methods['_calc_beta'] = getattr(self.module,
                phase_classnm+'beta')
            self._methods['_calc_K'] = getattr(self.module,
                phase_classnm+'K')
            self._methods['_calc_Kp'] = getattr(self.module,
                phase_classnm+'Kp')

            self._methods['_calc_mu'] = getattr(self.module,
                phase_classnm+'g')
            self._methods['_calc_a'] = getattr(self, 'not_coded')
            self._methods['_calc_da_dm'] = getattr(self, 'not_coded')

        self._exchange_equil = ExchangeEquil(self)

    def _init_pure_phase_props(self, phase_classnm):
        phase_obj = self.phase_obj

        if self.source == 'objc':
            element_comp = core.double_vector_to_array(
                phase_obj.formulaAsElementArray)
        else:
            method = getattr(self.module, phase_classnm+'elements')
            element_comp = method()
        atom_num = np.sum(element_comp)
        elem_num = element_comp.size
        mol_oxide_comp = chem.calc_mol_oxide_comp(element_comp)

        props = self.props
        props['endmember_name'] = np.array([str(props['phase_name'])])
        props['endmember_id'] = np.array([0]) #TK
        if self.source == 'objc':
            props['formula'] = np.array([str(phase_obj.phaseFormula)])
            props['molwt'] = np.array([phase_obj.mw])
            props['elemental_entropy'] = np.array(
                [phase_obj.entropyFromRobieEtAl1979])
        else:
            method = getattr(self.module, phase_classnm+'formula')
            props['formula'] = np.array([method()])
            method = getattr(self.module, phase_classnm+'mw')
            props['molwt'] = np.array([method()])
            props['elemental_entropy'] = np.array([None])
        props['atom_num'] = np.array([atom_num])
        props['element_comp'] = element_comp[np.newaxis,:]
        props['mol_oxide_comp'] = mol_oxide_comp[np.newaxis, :]

    @property
    def Berman_formula(self):
        """
        Representation of formula using Berman format

        Returns
        -------
        Chemical formula of phase (str)

        """
        return chem.get_Berman_formula(self.props['element_comp'])

    def not_coded(self, T, P, mol=None, V=None, param=None):
        raise AttributeError(
            'Function not implemented for this database model.')

    def _calc_G(self, T, P, mol=None, V=None):
        return (self._methods['_calc_G'])(T, P)

    def _calc_dG_dT(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dG_dT'])(T, P)

    def _calc_dG_dP(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dG_dP'])(T, P)

    def _calc_dG_dm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dG_dm'])(T, P)

    def _calc_dG_dw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_dG_dw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_dG_dw'])(T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d2G_dT2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2G_dT2'])(T, P)

    def _calc_d2G_dTdP(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2G_dTdP'])(T, P)

    def _calc_d2G_dTdm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2G_dTdm'])(T, P)

    def _calc_d2G_dTdw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d2G_dTdw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d2G_dTdw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d2G_dP2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2G_dP2'])(T, P)

    def _calc_d2G_dPdm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2G_dPdm'])(T, P)

    def _calc_d2G_dPdw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d2G_dPdw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d2G_dPdw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d2G_dm2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2G_dm2'])(T, P)

    def _calc_d2G_dmdw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d2G_dmdw'])(T, P, param)

    def _calc_d3G_dT3(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dT3'])(T, P)

    def _calc_d3G_dT2dP(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dT2dP'])(T, P)

    def _calc_d3G_dT2dm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dT2dm'])(T, P)

    def _calc_d3G_dT2dw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d3G_dT2dw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d3G_dT2dw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dTdP2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dTdP2'])(T, P)

    def _calc_d3G_dTdPdm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dTdPdm'])(T, P)

    def _calc_d3G_dTdPdw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d3G_dTdPdw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d3G_dTdPdw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dTdm2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dTdm2'])(T, P)

    def _calc_d3G_dTdmdw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d3G_dTdmdw'])(T, P, param)

    def _calc_d3G_dP3(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dP3'])(T, P)

    def _calc_d3G_dP2dm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dP2dm'])(T, P)

    def _calc_d3G_dP2dw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d3G_dP2dw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d3G_dP2dw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dPdm2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dPdm2'])(T, P)

    def _calc_d3G_dPdmdw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d3G_dPdmdw'])(T, P, param)

    def _calc_d3G_dm3(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d3G_dm3'])(T, P)

    def _calc_d3G_dm2dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d3G_dm2dw'])(T, P, param)

    def _calc_d4G_dT2dmdw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d4G_dT2dmdw'])(T, P, param)

    def _calc_d4G_dTdPdmdw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d4G_dTdPdmdw'])(T, P, param)

    def _calc_d4G_dTdm2dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d4G_dTdm2dw'])(T, P, param)

    def _calc_d4G_dP2dmdw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d4G_dP2dmdw'])(T, P, param)

    def _calc_d4G_dPdm2dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d4G_dPdm2dw'])(T, P, param)

    def _calc_d4G_dm3dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d4G_dm3dw'])(T, P, param)

    def _calc_d4G_dT3dw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d4G_dT3dw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dT3dw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dT2dPdw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d4G_dT2dPdw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dT2dPdw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dTdP2dw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d4G_dTdP2dw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dTdP2dw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dP3dw(self, T, P, mol=None, param=None):
        result = []
        if self.source == 'objc':
            iresult_arr = (self._methods['_calc_d4G_dP3dw'])(T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dP3dw'])(
                    T, P, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d5G_dT2dm2dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d5G_dT2dm2dw'])(T, P, param)

    def _calc_d5G_dTdPdm2dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d5G_dTdPdm2dw'])(T, P, param)

    def _calc_d5G_dP2dm2dw(self, T, P, mol=None, param=None):
        return (self._methods['_calc_d5G_dP2dm2dw'])(T, P, param)

    def _calc_H(self, T, P, mol=None, V=None):
        return (self._methods['_calc_H'])(T, P)

    def _calc_S(self, T, P, mol=None, V=None):
        return (self._methods['_calc_S'])(T, P)

    def _calc_dS_dm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dS_dm'])(T, P)

    def _calc_d2S_dm2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2S_dm2'])(T, P)

    def _calc_Cv(self, T, P, mol=None, V=None):
        return (self._methods['_calc_Cv'])(T, P)

    def _calc_Cp(self, T, P, mol=None, V=None):
        return (self._methods['_calc_Cp'])(T, P)

    def _calc_dCp_dT(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dCp_dT'])(T, P)

    def _calc_dCp_dm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dCp_dm'])(T, P)

    def _calc_V(self, T, P, mol=None, V=None):
        return (self._methods['_calc_V'])(T, P)

    def _calc_dV_dT(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dV_dT'])(T, P)

    def _calc_d2V_dmdT(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2V_dmdT'])(T, P)

    def _calc_d2V_dmdP(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2V_dmdP'])(T, P)

    def _calc_dV_dP(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dV_dP'])(T, P)

    def _calc_dV_dm(self, T, P, mol=None, V=None):
        return (self._methods['_calc_dV_dm'])(T, P)

    def _calc_d2V_dT2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2V_dT2'])(T, P)

    def _calc_d2V_dTdP(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2V_dTdP'])(T, P)

    def _calc_d2V_dP2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2V_dP2'])(T, P)

    def _calc_d2V_dm2(self, T, P, mol=None, V=None):
        return (self._methods['_calc_d2V_dm2'])(T, P)

    def _calc_density(self, T, P, mol=None, V=None):
        return (self._methods['_calc_density'])(T, P)

    def _calc_alpha(self, T, P, mol=None, V=None):
        return (self._methods['_calc_alpha'])(T, P)

    def _calc_beta(self, T, P, mol=None, V=None):
        return (self._methods['_calc_beta'])(T, P)

    def _calc_K(self, T, P, mol=None, V=None):
        return (self._methods['_calc_K'])(T, P)

    def _calc_Kp(self, T, P, mol=None, V=None):
        return (self._methods['_calc_Kp'])(T, P)

    # Compositional quantities
    def _calc_mu(self, T, P, mol=None, V=None, endmember=None, species=False):
        return (self._methods['_calc_G'])(T, P)

    def _calc_a(self, T, P, mol=None, V=None, endmember=None):
        return (self._methods['_calc_a'])(T, P)

    def _calc_da_dm(self, T, P, mol=None, V=None, endmember=None):
        return (self._methods['_calc_da_dm'])(T, P)

#===================================================
class SolutionPhase(Phase):
    """
    Solid solution phases.

    Implements the Phase interface.

    Parameters
    ----------
    phase_classnm : str
        Official class name for phase (implemented in src code).
    abbrev : str
        Official abbreviation of phase (regardless of implementation).
    calib : bool, default True
        Indicates whether sample phase should be calibration ready.

    Attributes
    ----------
    Attributes for this class are inherited from the Phase class.

    Notes
    -----
    * This code is highly dependent on implementation and is likely to change
      dramatically with changes in the underlying code that calculates phase
      properties.
    * The pure water phase, "H2O", is very complex and thus not available for
      calibration. The water phase will force its calib flag to False, regardless
      of input value.

    """
    def __init__(self, phase_classnm, abbrev, XTOL=1e-12, calib=True,
                 source='objc', coder_module=None):
        super().__init__(phase_classnm, abbrev, calib=calib, source=source,
                         coder_module=coder_module)
        self._XTOL = XTOL
        self._init_endmember_props()
        self._init_species()
        self._phase_type = 'solution'

        self._methods = {}
        if self.source == 'objc':
            self._methods['_calc_G'] = getattr(self._phase_obj,
                'getGibbsFreeEnergyFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dG_dT'] = getattr(self, 'not_coded')
            self._methods['_calc_dG_dP'] = getattr(self._phase_obj,
                'getVolumeFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dG_dm'] = getattr(self._phase_obj,
                'getDgDmFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dG_dw'] = getattr(self._phase_obj,
                'getDgDwFromMolesOfComponents_andT_andP_'
                ) if self.calib else getattr(self, 'not_coded')

            self._methods['_calc_d2G_dT2'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dTdP'] = getattr(self._phase_obj,
                'getDvDtFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2G_dTdm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dTdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dP2'] = getattr(self._phase_obj,
                'getDvDpFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2G_dPdm'] = getattr(self._phase_obj,
                'getDvDmFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2G_dPdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d2G_dm2'] = getattr(self._phase_obj,
                'getD2gDm2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2G_dmdw'] = getattr(self._phase_obj,
                'getChemicalPotentialDerivativesForParameter_usingMolesOfComponents_andT_andP_'
                ) if self.calib else getattr(self, 'not_coded')

            self._methods['_calc_d3G_dT3'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dT2dP'] = getattr(self._phase_obj,
                'getD2vDt2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dT2dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dT2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdP2'] = getattr(self._phase_obj,
                'getD2vDtDpFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dTdPdm'] = getattr(self._phase_obj,
                'getD2vDmDtFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dTdPdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdm2'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dTdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dP3'] = getattr(self._phase_obj,
                'getD2vDp2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dP2dm'] = getattr(self._phase_obj,
                'getD2vDmDpFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dP2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dPdm2'] = getattr(self._phase_obj,
                'getD2vDm2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dm3'] = getattr(self._phase_obj,
                'getD3gDm3FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d3G_dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dP2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dm3dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dPdm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT3dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dT2dPdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdP2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dP3dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d5G_dT2dm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dTdPdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dP2dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_H'] = getattr(self._phase_obj,
                'getEnthalpyFromMolesOfComponents_andT_andP_')
            self._methods['_calc_S'] = getattr(self._phase_obj,
                'getEntropyFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dS_dm'] = getattr(self._phase_obj,
                'getDsDmFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2S_dm2'] = getattr(self._phase_obj,
                'getD2sDm2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_V'] = getattr(self._phase_obj,
                'getVolumeFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dV_dT'] = getattr(self._phase_obj,
                'getDvDtFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dV_dP'] = getattr(self._phase_obj,
                'getDvDpFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dV_dm'] = getattr(self._phase_obj,
                'getDvDmFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2V_dT2'] = getattr(self._phase_obj,
                'getD2vDt2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2V_dTdP'] = getattr(self._phase_obj,
                'getD2vDtDpFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2V_dP2'] = getattr(self._phase_obj,
                'getD2vDp2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2V_dmdT'] = getattr(self._phase_obj,
                'getD2vDmDtFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2V_dmdP'] = getattr(self._phase_obj,
                'getD2vDmDpFromMolesOfComponents_andT_andP_')
            self._methods['_calc_d2V_dm2'] = getattr(self._phase_obj,
                'getD2vDm2FromMolesOfComponents_andT_andP_')
            self._methods['_calc_Cv'] = getattr(self, 'not_coded')
            self._methods['_calc_Cp'] = getattr(self._phase_obj,
                'getHeatCapacityFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dCp_dT'] = getattr(self._phase_obj,
                'getDcpDtFromMolesOfComponents_andT_andP_')
            self._methods['_calc_dCp_dm'] = getattr(self._phase_obj,
                'getDCpDmFromMolesOfComponents_andT_andP_')

            self._methods['_calc_density'] = getattr(self, 'not_coded')
            self._methods['_calc_alpha'] = getattr(self, 'not_coded')
            self._methods['_calc_beta'] = getattr(self, 'not_coded')
            self._methods['_calc_K'] = getattr(self, 'not_coded')
            self._methods['_calc_Kp'] = getattr(self, 'not_coded')

            self._methods['_calc_mu'] = getattr(self._phase_obj,
                'getChemicalPotentialFromMolesOfComponents_andT_andP_')
            self._methods['_calc_a'] = getattr(self._phase_obj,
                'getActivityFromMolesOfComponents_andT_andP_')
            self._methods['_calc_da_dm'] = getattr(self._phase_obj,
                'getDaDmFromMolesOfComponents_andT_andP_')

        else:
            phase_calibnm = phase_classnm.replace('calib', 'dparam') if (
                self.calib) else ''
            self._methods['_calc_G'] = getattr(self.module,
                phase_classnm+'g')
            self._methods['_calc_dG_dT'] = getattr(self.module,
                phase_classnm+'dgdt')
            self._methods['_calc_dG_dP'] = getattr(self.module,
                phase_classnm+'dgdp')
            self._methods['_calc_dG_dm'] = getattr(self.module,
                phase_classnm+'dgdn')
            self._methods['_calc_dG_dw'] = getattr(self.module,
                phase_calibnm+'g') if self.calib else getattr(
                self, 'not_coded')

            self._methods['_calc_d2G_dT2'] = getattr(self.module,
                phase_classnm+'d2gdt2')
            self._methods['_calc_d2G_dTdP'] = getattr(self.module,
                phase_classnm+'d2gdtdp')
            self._methods['_calc_d2G_dTdm'] = getattr(self.module,
                phase_classnm+'d2gdndt')
            self._methods['_calc_d2G_dTdw'] = getattr(self.module,
                phase_calibnm+'dgdt') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d2G_dP2'] = getattr(self.module,
                phase_classnm+'d2gdp2')
            self._methods['_calc_d2G_dPdm'] = getattr(self.module,
                phase_classnm+'d2gdndp')
            self._methods['_calc_d2G_dPdw'] = getattr(self.module,
                phase_calibnm+'dgdp') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d2G_dm2'] = getattr(self.module,
                phase_classnm+'d2gdn2')
            self._methods['_calc_d2G_dmdw'] = getattr(self.module,
                phase_calibnm+'dgdn') if self.calib else getattr(
                self, 'not_coded')

            self._methods['_calc_d3G_dT3'] = getattr(self.module,
                phase_classnm+'d3gdt3')
            self._methods['_calc_d3G_dT2dP'] = getattr(self.module,
                phase_classnm+'d3gdt2dp')
            self._methods['_calc_d3G_dT2dm'] = getattr(self.module,
                phase_classnm+'d3gdndt2')
            self._methods['_calc_d3G_dT2dw'] = getattr(self.module,
                phase_calibnm+'d2gdt2') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d3G_dTdP2'] = getattr(self.module,
                phase_classnm+'d3gdtdp2')
            self._methods['_calc_d3G_dTdPdm'] = getattr(self.module,
                phase_classnm+'d3gdndtdp')
            self._methods['_calc_d3G_dTdPdw'] = getattr(self.module,
                phase_calibnm+'d2gdtdp') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d3G_dTdm2'] = getattr(self.module,
                phase_classnm+'d3gdn2dt')
            self._methods['_calc_d3G_dTdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dP3'] = getattr(self.module,
                phase_classnm+'d3gdp3')
            self._methods['_calc_d3G_dP2dm'] = getattr(self.module,
                phase_classnm+'d3gdndp2')
            self._methods['_calc_d3G_dP2dw'] = getattr(self.module,
                phase_calibnm+'d2gdp2') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d3G_dPdm2'] = getattr(self.module,
                phase_classnm+'d3gdn2dp')
            self._methods['_calc_d3G_dPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d3G_dm3'] = getattr(self.module,
                phase_classnm+'d3gdn3')
            self._methods['_calc_d3G_dm2dw'] = getattr(self, 'not_coded')

            self._methods['_calc_d4G_dT2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdPdmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dP2dmdw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dTdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d4G_dPdm2dw'] = getattr(self, 'not_coded')

            # d4gdndt3, d4gdndt2dp, d4gdndtdp2, d4gdndp3 are defined in code
            # d4gdn2dt2, d4gdn2dtdp, d4gdn2dp2 are defined in code
            # d4gdn3dt, d4gdn3dp are defined in code

            self._methods['_calc_d4G_dT3dw'] = getattr(self.module,
                phase_calibnm+'d3gdt3') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dT2dPdw'] = getattr(self.module,
                phase_calibnm+'d3gdt2dp') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dTdP2dw'] = getattr(self.module,
                phase_calibnm+'d3gdtdp2') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dP3dw'] = getattr(self.module,
                phase_calibnm+'d3gdp3') if self.calib else getattr(
                self, 'not_coded')
            self._methods['_calc_d4G_dm3dw'] = getattr(self, 'not_coded')

            # d5gdn2dt3, d5gdn2dt2dp, d5gdn2dtdp2, d5gdn2dp3 are defined in code
            # d5gdn3dt2, d5gdn3dtdp, d5gdn3dp2 are defined in code

            self._methods['_calc_d5G_dT2dm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dTdPdm2dw'] = getattr(self, 'not_coded')
            self._methods['_calc_d5G_dP2dm2dw'] = getattr(self, 'not_coded')

            # d6gdn3dt3, d6gdn3dt2dp, d6gdn3dtdp2, d6gdn3dp3 are defined in code

            self._methods['_calc_H'] = getattr(self, 'not_coded')
            self._methods['_calc_S'] = getattr(self.module,
                phase_classnm+'s')
            self._methods['_calc_dS_dm'] = getattr(self, 'not_coded')
            self._methods['_calc_d2S_dm2'] = getattr(self, 'not_coded')
            self._methods['_calc_V'] = getattr(self.module,
                phase_classnm+'v')
            self._methods['_calc_dV_dT'] = getattr(self.module,
                phase_classnm+'d2gdtdp')
            self._methods['_calc_dV_dP'] = getattr(self.module,
                phase_classnm+'d2gdp2')
            self._methods['_calc_dV_dm'] = getattr(self.module,
                phase_classnm+'d2gdndp')
            self._methods['_calc_d2V_dT2'] = getattr(self.module,
                phase_classnm+'d3gdt2dp')
            self._methods['_calc_d2V_dTdP'] = getattr(self.module,
                phase_classnm+'d3gdtdp2')
            self._methods['_calc_d2V_dP2'] = getattr(self.module,
                phase_classnm+'d3gdp3')
            self._methods['_calc_d2V_dmdT'] = getattr(self.module,
                phase_classnm+'d3gdndtdp')
            self._methods['_calc_d2V_dmdP'] = getattr(self.module,
                phase_classnm+'d3gdndp2')
            self._methods['_calc_d2V_dm2'] = getattr(self.module,
                phase_classnm+'d3gdn2dp')
            self._methods['_calc_Cv'] = getattr(self.module,
                phase_classnm+'cv')
            self._methods['_calc_Cp'] = getattr(self.module,
                phase_classnm+'cp')
            self._methods['_calc_dCp_dT'] = getattr(self.module,
                phase_classnm+'dcpdt')
            self._methods['_calc_dCp_dm'] = getattr(self, 'not_coded')

            self._methods['_calc_density'] = getattr(self, 'not_coded')
            self._methods['_calc_alpha'] = getattr(self.module,
                phase_classnm+'alpha')
            self._methods['_calc_beta'] = getattr(self.module,
                phase_classnm+'beta')
            self._methods['_calc_K'] = getattr(self.module,
                phase_classnm+'K')
            self._methods['_calc_Kp'] = getattr(self.module,
                phase_classnm+'Kp')

            self._methods['_calc_mu'] = getattr(self.module,
                phase_classnm+'dgdn')
            self._methods['_calc_a'] = getattr(self, 'not_coded')
            self._methods['_calc_da_dm'] = getattr(self, 'not_coded')


        self._exchange_equil = ExchangeEquil(self)

    def _init_species(self):
        props = self.props
        XTOL = self._XTOL

        elem_comp_spec = props['species_elms']
        elem_comp_endmem = props['element_comp']
        if elem_comp_spec.shape[1]!=107:
            # from IPython import embed;embed()
            Nspec = elem_comp_spec.shape[0]
            elem_comp_spec = np.hstack(
                (np.zeros(Nspec)[:,np.newaxis], elem_comp_spec))
            Nendmem = elem_comp_endmem.shape[0]
            elem_comp_endmem = np.hstack(
                (np.zeros(Nendmem)[:,np.newaxis], elem_comp_endmem))

        elem_mask = np.any(elem_comp_spec>0,axis=0)
        elem_stoic_spec = elem_comp_spec[:,elem_mask]

        elems = chem.PERIODIC_ORDER[elem_mask]
        species_names = props['species_name']
        species_elem_stoic = pd.DataFrame(
            elem_stoic_spec, columns=elems, index=species_names)

        endmem_elem_stoic = pd.DataFrame(
            elem_comp_endmem[:,elem_mask],
            columns=elems, index=self.endmember_names)

        species_stoic = []

        for iname, ielem_stoic in species_elem_stoic.iterrows():
            iresults = np.linalg.lstsq(
                endmem_elem_stoic.T, ielem_stoic, rcond=None)
            istoic = iresults[0]
            istoic[np.abs(istoic)<XTOL] = 0
            species_stoic.append(istoic)

        species_stoic = pd.DataFrame(
            species_stoic, columns=endmem_elem_stoic.index,
            index=species_elem_stoic.index)


        # outcome variables
        self._endmember_elem_comp = endmem_elem_stoic
        self._species_elem_comp = species_elem_stoic
        self._species_stoic = species_stoic
        self._species_stoic_T = species_stoic.T
        self._species_stoic_T_vals = species_stoic.T.values

    def _init_endmember_props(self):
        phase_obj = self.phase_obj
        props = self.props

        if self.source == 'objc':
            try:
                result = self.phase_obj.supportsParameterCalibration()
                self._calib = True
            except:
                self._calib = False

        if self.source == 'objc':
            try:
                endmember_num = phase_obj.numberOfSolutionComponents()
                oxide_num = phase_obj.numberOfSolutionSpecies()
                species_num = phase_obj.numberOfSolutionSpecies()
            except:
                endmember_num = 0
                oxide_num = 0
                species_num = 0

        else:
            phase_classnm = self._phase_cls
            method = getattr(self.module, phase_classnm+'endmember_number')
            endmember_num = method()
            oxide_num = method()
            method = getattr(self.module, phase_classnm+'species_number')
            species_num = method()
            print ('Solution phase code generated by the coder module ' +
                'does not yet provide information on solution species. ' +
                'Species are proxied by components.')

        names = []
        molwt = []
        mol_oxide_comp = []
        element_comp = []
        formula = []
        atom_num = []
        species = []
        species_elms = []

        if self.source == 'objc':
            try:
                for i in range(0, endmember_num):
                    iendmember = phase_obj.componentAtIndex_(i)
                    iendmember_name = str(iendmember.phaseName)
                    iendmember_formula = str(iendmember.phaseFormula)
                    imolwt = iendmember.mw

                    ielems = core.double_vector_to_array(
                        iendmember.formulaAsElementArray)
                    iatom_num = np.sum(ielems)
                    imol_oxide_comp = chem.calc_mol_oxide_comp(ielems)

                    names.append(iendmember_name)
                    molwt.append(imolwt)
                    element_comp.append(ielems)
                    mol_oxide_comp.append(imol_oxide_comp)
                    formula.append(iendmember_formula)
                    atom_num.append(iatom_num)

            except:
                print('Error loading endmember properties for '
                      + self._props['phase_name'])
        else:
            for i in range(0, endmember_num):
                method = getattr(self.module, phase_classnm+'endmember_name')
                iendmember_name = method(i)
                method = getattr(self.module, phase_classnm+'endmember_formula')
                iendmember_formula = method(i)
                method = getattr(self.module, phase_classnm+'endmember_mw')
                imolwt = method(i)
                method = getattr(self.module, phase_classnm+'endmember_elements')
                ielems = method(i)
                iatom_num = np.sum(ielems)
                imol_oxide_comp = chem.calc_mol_oxide_comp(ielems)

                names.append(iendmember_name)
                molwt.append(imolwt)
                element_comp.append(ielems)
                mol_oxide_comp.append(imol_oxide_comp)
                formula.append(iendmember_formula)
                atom_num.append(iatom_num)

        if self.source == 'objc':
            try:
                for i in range(0,species_num):
                    ispecies_name = str(phase_obj.nameOfSolutionSpeciesAtIndex_(i))
                    ispecies_elms_arr = (
                        phase_obj.elementalCompositionOfSpeciesAtIndex_(i))
                    ispecies_elms = core.double_vector_to_array(
                        ispecies_elms_arr)

                    species.append(ispecies_name)
                    species_elms.append(ispecies_elms)

            except:
                print('Error loading species properties for '
                      + self._props['phase_name'])
        else:
            for i in range(0,species_num):
                method = getattr(self.module, phase_classnm+'species_name')
                ispecies_name = method(i)
                method = getattr(self.module, phase_classnm+'species_elements')
                ispecies_elms = method(i)
                species.append(ispecies_name)
                species_elms.append(ispecies_elms)
            print ('Solution phase code generated by the coder module ' +
                'does not yet provide information on species properties. ' +
                'Species are proxied by components.')

        names = np.array(names)
        molwt = np.array(molwt)
        element_comp = np.array(element_comp)
        mol_oxide_comp = np.array(mol_oxide_comp)
        formula = np.array(formula)
        atom_num = np.array(atom_num)
        oxide_space = np.any(mol_oxide_comp, axis=0)
        species = np.array(species)
        species_elms = np.array(species_elms)

        # endmember_props = OrderedDict()
        props['endmember_name'] = names
        props['endmember_num'] = endmember_num
        props['endmember_id'] = np.arange(endmember_num)
        props['species_num'] = species_num
        props['formula'] = formula
        props['atom_num'] = atom_num
        props['molwt'] = molwt
        props['elemental_entropy'] = [None]
        props['element_comp'] = element_comp
        props['mol_oxide_comp'] = mol_oxide_comp
        props['oxide_space'] = oxide_space
        props['species_name'] = species
        props['species_elms'] = species_elms
        # self._endmember_props = endmember_props


    @property
    def endmember_elem_comp(self):
        """
        Endmember elemental composition (compact)

        Returns
        -------
        Dataframe (or matrix) with elemental composition (cols)
        for each endmember (rows). Only includes elements that are part of the
        solution phase.

        """
        return self._endmember_elem_comp

    @property
    def species_elem_comp(self):
        """
        Species elemental composition (compact)

        Returns
        -------
        Dataframe (or matrix) with elemental composition (cols)
        for each species(rows). Only includes elements that are part of the
        solution phase.

        """
        return self._species_elem_comp

    @property
    def species_stoic(self):
        """
        Species Stoichiometry

        Returns
        -------
        Dataframe (or matrix) with molar endmember stoichiometry (cols)
        for each species (rows). Expresses species in terms of independent
        endmembers.

        """
        return self._species_stoic

    @property
    def species_stoic_T(self):
        """
        Species Stoichiometry Transpose

        Returns
        -------
        Dataframe (or matrix) with molar endmember stoichiometry (rows)
        for each species (cols). Expresses species in terms of independent
        endmembers.

        """
        return self._species_stoic_T

    ###############
    ##  Methods  ##
    ###############
    def calc_endmember_comp(self, mol_oxide_comp, method='least_squares',
                            output_residual=False, normalize=False,
                            decimals=10):
        """
        Get fraction of each endmember given the composition.

        Parameters
        ----------
        mol_oxide_comp : double array
            Amounts of each oxide in standard order (defined in OXIDES)
        decimals : int, default 10
            Number of decimals to round result to
        method : str, default 'least_squares'
            Method used to convert oxide composition (in moles) to moles of
            endmembers 'intrinsic' is alternate method, hardcoded by the
            solution implementation

        Returns
        -------
        endmember_comp : double array
            Best-fit molar composition in terms of endmembers
        mol_oxide_comp_residual : double array
            Residual molar oxide composition


        Notes
        -----
        * Eventually, we may want the ability to calculate endmember comp. using
          a variety of methods for inputing composition:
              * kind : ['wt_oxide', 'mol_oxide', 'element']
                    Identifies how composition is defined.

        """
        assert mol_oxide_comp.size == chem.oxide_props['oxide_num'], (
            'Oxide composition array not standard length')

        if method == 'least_squares':
            mol_oxide_comp_endmembers = self.props['mol_oxide_comp']
            # NOTE: mol oxide composition is extensive (in absolute mols)
            # need not sum to 1.
            output = np.linalg.lstsq(
                mol_oxide_comp_endmembers.T, mol_oxide_comp, rcond=None)

            endmember_comp = output[0]
            endmember_comp = np.round(endmember_comp, decimals=decimals)

            mol_oxide_comp_model = np.dot(mol_oxide_comp_endmembers.T,
                endmember_comp)
            mol_oxide_comp_residual = (mol_oxide_comp - mol_oxide_comp_model)

        elif method == 'intrinsic':
            elems = chem.PERIODIC_ORDER
            monovalent_cations = chem.oxide_props['cations']
            monovalent_elem_ind = np.array(
                [np.where(elems==icat)[0][0] for icat in monovalent_cations])
            oxy_elem_ind = np.where(elems=='O')[0][0]
            moles_elm = np.zeros(chem.PERIODIC_ORDER.size)
            for i in range(0, mol_oxide_comp.size):
                ind = monovalent_elem_ind[i]
                nCat = chem.oxide_props['cat_num'][i]
                nOx = chem.oxide_props['oxy_num'][i]
                moles_elm[ind] += mol_oxide_comp[i]*nCat
                moles_elm[oxy_elem_ind] += mol_oxide_comp[i]*nOx
            if self.source == 'objc':
                moles_pot_arr = self._phase_obj.convertElementsToMoles_(
                    core.array_to_ctype_array(moles_elm))
                endmember_comp = core.double_vector_to_array(moles_pot_arr)
            else:
                method = getattr(self.module,
                    self._phase_cls+'conv_elm_to_moles')
                endmember_comp = method(moles_elm)
            mol_oxide_comp_residual = None

        else:
            assert False, (
                'Method argument is not valid. Choose either "least_squares" ' +
                'or "intrinic".')

        if normalize:
            mol_oxide_tot = np.sum(endmember_comp)
            endmember_comp /= mol_oxide_tot
            if mol_oxide_comp_residual is not None:
                mol_oxide_comp_residual /= mol_oxide_tot

        if output_residual:
            return endmember_comp, mol_oxide_comp_residual
        else:
            return endmember_comp

    def test_endmember_comp(self, mol_comp):
        """
        Tests validity of endmember component moles array

        Parameters
        ----------
        mol_comp : double array
            Mole numbers of each component in the solution

        Returns
        -------
        flag : boolean
            True is composition is valid, otherwise False.
        """
        if self.source == 'objc':
            result = self._phase_obj.testPermissibleValuesOfComponents_(
                core.array_to_ctype_array(mol_comp))
        else:
            method = getattr(self.module, self._phase_cls+'test_moles')
            result = method(mol_comp)
        return (True if result else False)

    def compute_formula(self, T, P, mol_comp):
        """
        Converts an input array of moles of endmember components to
        the chemical formula of the phase

        Parameters
        ----------
        T : double
            Temperature in Kelvins
        P : double
            Pressure in bars
        mol_comp : double array
            Mole numbers of each component in the solution

        Returns
        -------
        formula : str
            A string with the formula of the phase
        """
        if self.source == 'objc':
            return str(self._phase_obj.getFormulaFromMolesOfComponents_andT_andP_(
            core.array_to_ctype_array(mol_comp), T, P))
        else:
            phase_classnm = self._phase_cls
            method = getattr(self.module, phase_classnm+'formula')
            result = method(T, P, mol_comp)
            return result


    @deprecation.deprecated(
        deprecated_in="1.0", removed_in="2.0",
        details=("This legacy function name has a typo. "
                 "Use convert_endmember_comp instead.")
        )
    def covert_endmember_comp(self, mol_comp, output='total_moles'):

        return self.convert_endmember_comp(mol_comp, output=output)

    def convert_endmember_comp(self, mol_comp, output='total_moles'):
        """
        Converts an input array of moles of endmember components to
        the specified quantity

        Parameters
        ----------
        mol_comp : double array
            Mole numbers of each component in the solution
        output : str, default = 'total_moles'
            Output quantity:
              - 'total_moles' - double
              - 'moles_elements' - double array (standard order and length)
              - 'mole_fraction' - double array (same order and length as input)
              - 'moles_species' - pandas series or array w/ accepted order

        Returns
        -------
        result : double or double array/pandas series
            The computed quantity as double or double array (or series)
        """

        if output=='moles_species':
            species_stoic = self.species_stoic
            mol_species_inv, rnorm = optim.nnls(species_stoic.T, mol_comp)
            mol_species_inv = pd.Series(
                mol_species_inv, index=species_stoic.index)

            return mol_species_inv


        if self.source == 'objc':
            if output == 'total_moles':
                result = self._phase_obj.totalMolesFromMolesOfComponents_(
                    core.array_to_ctype_array(mol_comp))
                return result
            elif output == 'moles_elements':
                result = self._phase_obj.convertMolesToElements_(
                    core.array_to_ctype_array(mol_comp))
                return core.double_vector_to_array(result)
            elif output == 'mole_fraction':
                result = self._phase_obj.convertMolesToMoleFractions_(
                    core.array_to_ctype_array(mol_comp))
                return core.double_vector_to_array(result)
            else:
                print ("output option must be 'total_moles', " +
                    "'moles_elements', or 'mole_fraction'.")
                return np.empty(self.props['endmember_num'])
        else:
            phase_classnm = self._phase_cls
            if output == 'total_moles':
                method = getattr(self.module,
                    self._phase_cls+'conv_moles_to_tot_moles')
                return method(mol_comp)
            elif output == 'moles_elements':
                method = getattr(self.module,
                    self._phase_cls+'conv_moles_to_elm')
                return method(mol_comp)
            elif output == 'mole_fraction':
                method = getattr(self.module,
                    self._phase_cls+'conv_moles_to_mole_frac')
                return method(mol_comp)
            else:
                print ("output option must be 'total_moles', " +
                    "'moles_elements', or 'mole_fraction'.")
                return np.empty(self.props['endmember_num'])

    def convert_species_to_comp(self, mol_species, method='linear'):
        """
        Converts an input array of moles of species to moles of
        endmember components

        Parameters
        ----------
        mol_species : double array
            Mole numbers of each species in the solution

        method : {'linear', 'source'}
            Default is 'linear', using basic dot-product. Otherwise, optimized
            methods can be used from source code by selecting 'source'.

        Returns
        -------
        result : double array
            Moles of endmember components
        """
        if method=='linear':
            species_stoic_T_vals = self._species_stoic_T_vals
            # mol_endmem = species_stoic_T.dot(mol_species)
            mol_endmem = np.dot(species_stoic_T_vals,mol_species)
            return mol_endmem

        if self.source == 'objc':
            result_arr = self._phase_obj.convertMolesOfSpeciesToMolesOfComponents_(
                    core.array_to_ctype_array(mol_species))
            return core.double_vector_to_array(result_arr)
        else:
            print ('Solution phase code generated by the coder module ' +
                'does not yet provide a conversion method.')
            return np.empty(self.props['endmember_num'])

    def convert_elements(self, mol_elm, output='moles_end'):
        """
        Converts an array of mole numbers of elements (in the standard order) to
        the specified output quantity

        Parameters
        ----------
        mole_elm : double array
            Mole numbers of elements in the standard order
        output : str, default = 'moles_end'
            Output quantity:
              - 'moles_end' - double array of moles of endmembers
              - 'total_moles' - double, sum of moles of endmembers
              - 'total_grams' - double, sum of grams of solution

        Returns
        -------
        result : double or double array
            The computed quantity as double or double array
        """
        if self.source == 'objc':
            if output == 'moles_end':
                result = self._phase_obj.convertElementsToMoles_(
                    core.array_to_ctype_array(mol_elm))
                return core.double_vector_to_array(result)
            elif output == 'total_moles':
                result = self._phase_obj.convertElementsToTotalMoles_(
                    core.array_to_ctype_array(mol_elm))
                return result
            elif output == 'total_grams':
                result = self._phase_obj.convertElementsToTotalMass_(
                    core.array_to_ctype_array(mol_elm))
                return result
            else:
                print ("output option must be 'moles_end', 'total_moles', " +
                    "or 'total_grams'.")
            return None
        else:
            phase_classnm = self._phase_cls
            if output == 'moles_end':
                method = getattr(self.module, phase_classnm+'conv_elm_to_moles')
                return method(mol_elm)
            elif output == 'total_moles':
                method = getattr(self.module,
                    phase_classnm+'conv_elm_to_tot_moles')
                return method(mol_elm)
            elif output == 'total_grams':
                method = getattr(self.module,
                    phase_classnm+'conv_elm_to_tot_grams')
                return method(mol_elm)
            else:
                print ("output option must be 'moles_end', 'total_moles', " +
                    "or 'total_grams'.")
            return None

    def get_endmember_ind(self, mol_oxide_comp, get_endmember_comp=False,
                          TOL=1e-6):
        """
        Get index of endmember that best matches composition.

        Parameters
        ----------
        mol_oxide_comp : double array
            Amounts of each oxide in standard order (defined in OXIDES)
        TOL : double, default 1e-6
            Allowed tolerance for mismatch to defined composition
        get_endmember_comp : bool, default False
            If true, also return endmember composition array.

        Returns
        -------
        endmember_ind : int
            Index of best-fit endmember
        endmember_comp : double array, optional
            Composition array in terms of endmembers.

            Return if get_endmember_comp==True
        """
        endmember_comp, mol_oxide_comp_residual = (
            self.calc_endmember_comp(mol_oxide_comp))

        endmember_ind = np.argmax(endmember_comp)
        endmember_scl_comp = endmember_comp/endmember_comp[endmember_ind]

        ind_all = np.arange(endmember_comp.size)
        ind_else = np.delete(ind_all, endmember_ind)

        assert np.all(np.abs(
            endmember_scl_comp[ind_else] < TOL)), (
                'Provided composition does not match '
                'an endmember.'
            )

        if get_endmember_comp:
            return endmember_ind, endmember_comp
        else:
            return endmember_ind

    def not_coded(self, T, P, mol=None, V=None, param=None):
        raise AttributeError(
            'Function not implemented for this database model.')

    def _calc_G(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_G'])(cmol, T, P)
        else:
            result = (self._methods['_calc_G'])(T, P, mol)
        return result

    def _calc_dG_dT(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_dG_dT'])(cmol, T, P)
        else:
            result = (self._methods['_calc_dG_dT'])(T, P, mol)
        return result

    def _calc_dG_dP(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_dG_dP'])(cmol, T, P)
        else:
            result = (self._methods['_calc_dG_dP'])(T, P, mol)
        return result

    def _calc_dG_dm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_dG_dm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_dG_dm'])(T, P, mol)

        return result

    def _calc_dG_dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_dG_dw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_dG_dw'])(T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d2G_dT2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d2G_dT2'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d2G_dT2'])(T, P, mol)
        return result

    def _calc_d2G_dTdP(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d2G_dTdP'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d2G_dTdP'])(T, P, mol)
        return result

    def _calc_d2G_dTdm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2G_dTdm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d2G_dTdm'])(T, P, mol)

        return result

    def _calc_d2G_dTdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d2G_dTdw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d2G_dTdw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d2G_dP2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d2G_dP2'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d2G_dP2'])(T, P, mol)
        return result

    def _calc_d2G_dPdm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2G_dPdm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d2G_dPdm'])(T, P, mol)

        return result

    def _calc_d2G_dPdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d2G_dPdw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d2G_dPdw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d2G_dm2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2G_dm2'])(cmol, T, P)
            result = core.double_matrix_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d2G_dm2'])(T, P, mol)

        return result

    def _calc_d2G_dmdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                name = self._param_props['param_names'][iparam]
                cmol = core.array_to_ctype_array(mol)
                iresult_arr = (self._methods['_calc_d2G_dmdw'])(
                    name, cmol, T, P)
                iresult = core.double_vector_to_array(iresult_arr)
            else:
                iresult = (self._methods['_calc_d2G_dmdw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dT3(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d3G_dT3'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d3G_dT3'])(T, P, mol)
        return result

    def _calc_d3G_dT2dP(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d3G_dT2dP'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d3G_dT2dP'])(T, P, mol)
        return result

    def _calc_d3G_dT2dm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d3G_dT2dm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d3G_dT2dm'])(T, P, mol)

        return result

    def _calc_d3G_dT2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d3G_dT2dw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d3G_dT2dw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dTdP2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d3G_dTdP2'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d3G_dTdP2'])(T, P, mol)
        return result

    def _calc_d3G_dTdPdm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d3G_dTdPdm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d3G_dTdPdm'])(T, P, mol)

        return result

    def _calc_d3G_dTdPdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d3G_dTdPdw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d3G_dTdPdw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dTdm2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d3G_dTdm2'])(cmol, T, P)
            result = core.double_matrix_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d3G_dTdm2'])(T, P, mol)

        return result

    def _calc_d3G_dTdmdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_arr = (self._methods['_calc_d3G_dTdmdw'])(
                    iparam, cmol, T, P)
                iresult = core.double_vector_to_array(iresult_arr)
            else:
                iresult = (self._methods['_calc_d3G_dTdmdw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dP3(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d3G_dP3'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d3G_dP3'])(T, P, mol)
        return result

    def _calc_d3G_dP2dm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d3G_dP2dm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d3G_dP2dm'])(T, P, mol)

        return result

    def _calc_d3G_dP2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d3G_dP2dw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d3G_dP2dw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dPdm2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d3G_dPdm2'])(cmol, T, P)
            result = core.double_matrix_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d3G_dPdm2'])(T, P, mol)

        return result

    def _calc_d3G_dPdmdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_arr = (self._methods['_calc_d3G_dPdmdw'])(
                    iparam, cmol, T, P)
                iresult = core.double_vector_to_array(iresult_arr)
            else:
                iresult = (self._methods['_calc_d3G_dPdmdw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d3G_dm3(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d3G_dm3'])(cmol, T, P)
            result = core.double_tensor_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d3G_dm3'])(T, P, mol)

        return result

    def _calc_d3G_dm2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_mat = (self._methods['_calc_d3G_dm2dw'])(cmol, T, P)
                iresult = core.double_matrix_to_array(iresult_mat)
            else:
                iresult = (self._methods['_calc_d3G_dm2dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dT3dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d4G_dT3dw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dT3dw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dT2dPdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d4G_dT2dPdw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dT2dPdw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dT2dmdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_arr = (self._methods['_calc_d4G_dT2dmdw'])(
                    iparam, cmol, T, P)
                iresult = core.double_vector_to_array(iresult_arr)
            else:
                iresult = (self._methods['_calc_d4G_dT2dmdw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dTdP2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d4G_dTdP2dw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dTdP2dw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dTdPdmdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_arr = (self._methods['_calc_d4G_dTdPdmdw'])(
                    iparam, cmol, T, P)
                iresult = core.double_vector_to_array(iresult_arr)
            else:
                iresult = (self._methods['_calc_d4G_dTdPdmdw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dTdm2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_mat = (self._methods['_calc_d4G_dTdm2dw'])(cmol, T, P)
                iresult = core.double_matrix_to_array(iresult_mat)
            else:
                iresult = (self._methods['_calc_d4G_dTdm2dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dP3dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_arr = (self._methods['_calc_d4G_dP3dw'])(cmol, T, P)
            iresult = core.double_vector_to_array(iresult_arr)
        for iparam in param:
            if self.source == 'objc':
                iresult_term = iresult[iparam]
            else:
                iresult_term = (self._methods['_calc_d4G_dP3dw'])(
                    T, P, mol, iparam)
            result.append(iresult_term)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dP2dmdw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_arr = (self._methods['_calc_d4G_dP2dmdw'])(
                    iparam, cmol, T, P)
                iresult = core.double_vector_to_array(iresult_arr)
            else:
                iresult = (self._methods['_calc_d4G_dP2dmdw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dPdm2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_mat = (self._methods['_calc_d4G_dPdm2dw'])(cmol, T, P)
                iresult = core.double_matrix_to_array(iresult_mat)
            else:
                iresult = (self._methods['_calc_d4G_dPdm2dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d4G_dm3dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d4G_dm3dw'])(cmol, T, P)
            iresult = core.double_tensor_to_array(iresult_mat)
        for iparam in param:
            if self.source == 'objc':
                iresult = iresult[iparam]
            else:
                iresult = (self._methods['_calc_d4G_dm3dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d5G_dT2dm2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_mat = (self._methods['_calc_d5G_dT2dm2dw'])(cmol, T, P)
                iresult = core.double_matrix_to_array(iresult_mat)
            else:
                iresult = (self._methods['_calc_d5G_dT2dm2dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d5G_dTdPdm2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_mat = (self._methods['_calc_d5G_dTdPdm2dw'])(cmol, T, P)
                iresult = core.double_matrix_to_array(iresult_mat)
            else:
                iresult = (self._methods['_calc_d5G_dTdPdm2dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_d5G_dP2dm2dw(self, T, P, mol=None, param=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        result = []
        for iparam in param:
            if self.source == 'objc':
                cmol = core.array_to_ctype_array(mol)
                iresult_mat = (self._methods['_calc_d5G_dP2dm2dw'])(cmol, T, P)
                iresult = core.double_matrix_to_array(iresult_mat)
            else:
                iresult = (self._methods['_calc_d5G_dP2dm2dw'])(T, P, mol, iparam)
            result.append(iresult)

        result = np.array(result) if len(result) > 1 else result[0]
        return result

    def _calc_H(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_H'])(cmol, T, P)
        else:
            result = (self._methods['_calc_G'])(T, P, mol) + T*(
                self._methods['_calc_S'])(T, P, mol)
        return result

    def _calc_S(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_S'])(cmol, T, P)
        else:
            result = (self._methods['_calc_S'])(T, P, mol)
        return result

    def _calc_V(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_V'])(cmol, T, P)
        else:
            result = (self._methods['_calc_V'])(T, P, mol)
        return result

    def _calc_mu(self, T, P, mol=None, V=None, endmember=None, species=False):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            chem_pot = []
            cmol = core.array_to_ctype_array(mol)
            for iendmem in endmember:
                if species:
                    ichem_pot_arr = (
                        self._phase_obj.
                        chemicalPotentialsOfSpeciesFromMolesOfComponents_andT_andP_(
                            cmol, T, P)
                    )
                else:
                    ichem_pot_arr = (self._methods['_calc_mu'])(cmol, T, P)

                ichem_pot = core.double_vector_to_array(ichem_pot_arr)
                if iendmem >= 0:
                    ichem_pot = ichem_pot[int(iendmem)]

                chem_pot.append(ichem_pot)
            chem_pot = np.array(chem_pot) if endmember[0] == -1 else chem_pot[0]
        else:
            chem_pot = (self._methods['_calc_mu'])(T, P, mol)
            if endmember is not None:
                result = []
                for iendmem in endmember:
                    result.append(chem_pot[iendmem])
                chem_pot = np.array(result)

        return chem_pot

    def _calc_a(self, T, P, mol=None, V=None, endmember=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            chem_pot = []
            cmol = core.array_to_ctype_array(mol)
            for iendmem in endmember:
                ichem_pot_arr = ((self._methods['_calc_a'])(cmol, T, P))
                ichem_pot = core.double_vector_to_array(ichem_pot_arr)
                if iendmem >= 0:
                    ichem_pot = ichem_pot[int(iendmem)]

                chem_pot.append(ichem_pot)
            chem_pot = np.array(chem_pot) if endmember[0] == -1 else chem_pot[0]
        else:
            chem_pot = (self._methods['_calc_a'])(T, P, mol)

        return chem_pot

    def _calc_dS_dm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_dS_dm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = -(self._methods['_calc_d2G_dTdm'])(T, P, mol)

        return result

    def _calc_dV_dm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_dV_dm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_dV_dm'])(T, P, mol)

        return result

    def _calc_da_dm(self, T, P, mol=None, V=None, endmember=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_da_dm'])(cmol, T, P)
            iresult = core.double_matrix_to_array(iresult_mat)
            if endmember >= 0:
                result = []
                for iendmem in endmember:
                    iresult = iresult[[int(iendmem)], :][0]
                    result.append(iresult)
                result = result[0]
            else:
                result = iresult
        else:
            result = (self._methods['_calc_da_dm'])(T, P, mol)
        return result

    def _calc_d2S_dm2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2S_dm2'])(cmol, T, P)
            result = core.double_matrix_to_array(iresult_mat)
        else:
            result = -(self._methods['_calc_d3G_dTdm2'])(T, P, mol)

        return result

    def _calc_d2V_dm2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2V_dm2'])(cmol, T, P)
            result = core.double_matrix_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d2V_dm2'])(T, P, mol)

        return result

    def _calc_Cv(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )
        return (self._methods['_calc_Cv'])(T, P, mol)

    def _calc_Cp(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for '
            'SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_Cp'])(cmol, T, P)
        else:
            result = (self._methods['_calc_Cp'])(T, P, mol)

        return result

    def _calc_dV_dT(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_dV_dT'])(cmol, T, P)
        else:
            result = (self._methods['_calc_dV_dT'])(T, P, mol)

        return result

    def _calc_dV_dP(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_dV_dP'])(cmol, T, P)
        else:
            result = (self._methods['_calc_dV_dP'])(T, P, mol)

        return result

    def _calc_dCp_dT(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_dCp_dT'])(cmol, T, P)
        else:
            result = (self._methods['_calc_dCp_dT'])(T, P, mol)

        return result

    def _calc_dCp_dm(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_dCp_dm'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = -(self._methods['_calc_d3G_dT2dm'])(T, P, mol)/T

        return result

    def _calc_d2V_dT2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d2V_dT2'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d2V_dT2'])(T, P, mol)

        return result

    def _calc_d2V_dTdP(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d2V_dTdP'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d2V_dTdP'])(T, P, mol)

        return result

    def _calc_d2V_dP2(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_d2V_dP2'])(cmol, T, P)
        else:
            result = (self._methods['_calc_d2V_dP2'])(T, P, mol)

        return result

    def _calc_d2V_dmdT(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2V_dmdT'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d2V_dmdT'])(T, P, mol)

        return result

    def _calc_d2V_dmdP(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            iresult_mat = (self._methods['_calc_d2V_dmdP'])(cmol, T, P)
            result = core.double_vector_to_array(iresult_mat)
        else:
            result = (self._methods['_calc_d2V_dmdP'])(T, P, mol)

        return result

    def _calc_density(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_density'])(cmol, T, P)
        else:
            result = (self._methods['_calc_density'])(T, P, mol)

        return result

    def _calc_alpha(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_alpha'])(cmol, T, P)
        else:
            result = (self._methods['_calc_alpha'])(T, P, mol)

        return result

    def _calc_beta(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_beta'])(cmol, T, P)
        else:
            result = (self._methods['_calc_beta'])(T, P, mol)

        return result

    def _calc_K(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_K'])(cmol, T, P)
        else:
            result = (self._methods['_calc_K'])(T, P, mol)

        return result

    def _calc_Kp(self, T, P, mol=None, V=None):
        assert mol is not None, (
            'Molar composition "mol" must be defined for SolutionPhase.'
        )

        if self.source == 'objc':
            cmol = core.array_to_ctype_array(mol)
            result = (self._methods['_calc_Kp'])(cmol, T, P)
        else:
            result = (self._methods['_calc_Kp'])(T, P, mol)

        return result

#===================================================


# class Sample:
#     """
#     Sample Phase with specific properties.
#
#     Parameters
#     ----------
#     phase_classnm : str
#         Official class name for phase (implemented in src code).
#     abbrev : str
#         Official abbreviation of phase (regardless of implementation).
#     mol : array, optional
#         Molar composition vector describing amount of each endmember. This
#         quantity is only meaningful for solution phases, since pure phases have
#         fixed composition (default is an empty list).
#     kind : {'Pure','Solution','Aqueous','Gas'}
#         Defines kind of sample phase.
#     calib : bool, default True
#         Indicates whether sample phase should be calibration ready.
#
#
#     Attributes
#     ----------
#     mol
#     kind
#     phase : Phase object
#         Stores implementation of phase, including all property parameter values.
#
#     """
#
#     def __init__(self, phase_classnm, abbrev, mol=None,
#                  kind='Pure', calib=True):
#
#         self.mol = mol
#         self.kind = kind
#
#         if kind == 'Pure':
#             phase = PurePhase(phase_classnm, abbrev, calib=calib)
#         elif kind == 'Solution':
#             phase = SolutionPhase(phase_classnm, abbrev, calib=calib)
#         elif kind == 'Aqueous':
#             phase = AqueousPhase( phase_classnm, abbrev, calib=calib )
#         elif kind == 'Gas':
#             phase = GasPhase( phase_classnm, abbrev, calib=calib )
#         else:
#             raise NotImplemented('Sample kind "'+kind+'" is not yet implimented')
#
#         self.phase = phase
#
#         pass
#===================================================
    # def get_dG_dw( self, mol_a, T, P ):
    #     assert False, '
    #     mol_vec = core.array_to_double_vector(mol_a)
    #     print(mol_vec)
    #     dG_dw_vec = self._phase_obj.getDgDwFromMolesOfComponents_andT_andP_( mol_vec, T, P )
    #     print(dG_dw_vec)
    #     dG_dw_a = core.double_vector_to_array( dG_dw_vec )

    #     return dG_dw_a

    # def get_dchempot_dw
    # getChemicalPotentialDerivativesForParameterArray:(NSArray *)array usingMolesOfComponents:(double *)m andT:(double)t andP:

    # def get_dchempot_dw( self, T_a, P_a ):
    #     self._phase_obj.getChemicalPotentialDerivativesForParameter_usingMolesOfComponents_andT_andP_('S',m,T,P)
class ExchangeEquil():
    phase: Type[Phase]
    _phase: Type[Phase]
    def __init__(self, phase: Type[Phase], X_min: float = 1e-10,
                 ATOL: float =1e-3, LOGXTOL: float =1e-3, iter_max: int =50):
        self._phase = phase
        self._X_min = X_min
        self._ATOL = ATOL
        self._LOGXTOL = LOGXTOL
        self._iter_max = iter_max

        # Switch to dependent species space
        # Nc_full = phase.endmember_num
        if phase.endmember_num==1:
            Nc_full = 1
        else:
            Nc_full = phase.species_stoic.shape[0]
        # print(phase.endmember_names)
        # print('Nc_full', Nc_full)

        self.Nc_full = Nc_full

        self.endmem_num = phase.endmember_num
        self.muM = np.zeros(Nc_full)
        self.phi = np.zeros(Nc_full)
        self.phi_prev = np.zeros(Nc_full)
        # print('phi = ', self.phi)

        self.site_m = np.nan



        X = np.full(Nc_full, X_min)
        X_prev = np.full(Nc_full, X_min)

        self.X = X
        self.X_prev = X_prev
        # print('X = ', self.X)

        self.X_mask = np.full(Nc_full,True)


        source = phase.source

        if source=='coder':
            chempot_fun = lambda T, P, mol=None: phase.gibbs_energy(
                T, P, mol=mol, deriv={'dmol':1})
        else:
            chempot_fun = phase.chem_potential


        def chempot_update(T, P, mol=None, phase=phase, deriv=False,
                           chempot_fun=chempot_fun):

            species_stoic = phase.species_stoic.values
            species_stoic_T = phase.species_stoic_T.values
            mol_species = mol
            mol_endmems = phase.convert_species_to_comp(mol_species)

            mu_endmems = phase.chem_potential(
                T, P, mol=mol_endmems).squeeze()

            mu_endmems = chempot_fun(T, P, mol=mol_endmems).squeeze()
            # mu_species = phase.species_stoic.dot(mu_endmems).values
            mu_species = np.dot(species_stoic, mu_endmems)

            if deriv:
                dmudn_vals = phase.gibbs_energy(
                    T, P, mol=mol_endmems, deriv={'dmol':2}).squeeze()

                # dmudn = pd.DataFrame(
                #     dmudn_vals, columns=phase.endmember_names,
                #     index=phase.endmember_names)

                # dmudn_species = phase.species_stoic.dot(
                #     dmudn).dot(phase.species_stoic.T).values

                dmudn_species = np.dot(
                    np.dot(species_stoic, dmudn_vals),species_stoic_T)
                return mu_species, dmudn_species
            else:
                return (mu_species, )

        self.chempot_update = chempot_update


        # if source=='coder':
        #     def chempot_update(T, P, mol=None, phase=phase, deriv=False):
        #         mol_species = mol
        #         mol_endmems = phase.convert_species_to_comp(mol_species)
        #         mu_endmems = phase.gibbs_energy(
        #             T, P, mol=mol_endmems, deriv={'dmol':1}).squeeze()
        #         mu_species = phase.species_stoic.dot(mu_endmems).values
        #
        #         if deriv:
        #             dmudn_vals = phase.gibbs_energy(
        #                 T, P, mol=mol_endmems, deriv={'dmol':2}).squeeze()
        #
        #             dmudn = pd.DataFrame(
        #                 dmudn_vals, columns=phase.endmember_names,
        #                 index=phase.endmember_names)
        #
        #             ### FIX
        #             assert False, 'fix this!'
        #             dmudn_species = phase.species_stoic.dot(
        #                 dmudn).dot(phase.species_stoic.T)
        #
        #             return mu_species, dmudn_species.values
        #
        #         else:
        #             return (mu_species,)
        #
        #     self.chempot_update = chempot_update
        # else:
        #     def chempot_update(T, P, mol=None, phase=phase, deriv=False):
        #
        #         species_stoic = phase.species_stoic.values
        #         species_stoic_T = phase.species_stoic_T.values
        #         mol_species = mol
        #         mol_endmems = phase.convert_species_to_comp(mol_species)
        #
        #         mu_endmems = phase.chem_potential(
        #             T, P, mol=mol_endmems).squeeze()
        #         # mu_species = phase.species_stoic.dot(mu_endmems).values
        #         mu_species = np.dot(species_stoic, mu_endmems)
        #
        #         if deriv:
        #             dmudn_vals = phase.gibbs_energy(
        #                 T, P, mol=mol_endmems, deriv={'dmol':2}).squeeze()
        #
        #             # dmudn = pd.DataFrame(
        #             #     dmudn_vals, columns=phase.endmember_names,
        #             #     index=phase.endmember_names)
        #
        #             # dmudn_species = phase.species_stoic.dot(
        #             #     dmudn).dot(phase.species_stoic.T).values
        #
        #             dmudn_species = np.dot(
        #                 np.dot(species_stoic, dmudn_vals),species_stoic_T)
        #             return mu_species, dmudn_species
        #         else:
        #             return (mu_species, )
        #
        #     # self.chempot_update = phase.chem_potential
        #     self.chempot_update = chempot_update


    @property
    def phase(self):
        return self._phase

    @property
    def ATOL(self):
        return self._ATOL

    @property
    def LOGXTOL(self):
        return self._LOGXTOL

    @property
    def iter_max(self):
        return self._iter_max

    @property
    def X_min(self):
        return self._X_min

    ###########
    def _est_site_mult(self, t, p):
        """
        Estimates site multiplicity of a solution phase
        """

        X_min = self.X_min
        endmem_num = self.endmem_num


        phs = self.phase
        x_ave = 1.0/endmem_num
        s_mid = phs.entropy(t, p, mol=np.full(endmem_num, x_ave))
        s_idl = -8.3143*np.log(x_ave)
        s = s_mid
        x_end = np.full(endmem_num, X_min)
        for i in range(0,endmem_num):
            x_end[i] = 0.99999998
            s -= x_ave*phs.entropy(t, p, mol=x_end)
            x_end[i] = X_min
        return np.rint(s/s_idl)

    def _tailored_affinity_and_comp(self, t, p, mu, debug=False):
        """
        Use a specific affinity and composition solver if available
        """
        phs = self.phase
        if phs.identifier == 'Objective-C-base':
            method = getattr(phs.phase_obj,
                'affinityAndCompositionFromLiquidChemicalPotentialSum_andT_andP_', None)
            if method:
                cmu = core.array_to_ctype_array(mu)
                if debug > 0:
                    print ('Calling tailored Affinity and Comp routine for',
                       phs.class_name)
                result = method(cmu, t, p)
                if sys.platform == "darwin":
                    A = float(result.objectAtIndex_(0).doubleValue)
                else:
                    A = float(result.objectAtIndex_(0))
                X = np.zeros(mu.size)
                for i in range(0,mu.size):
                    if sys.platform == "darwin":
                        X[i] = float(result.objectAtIndex_(i+1).doubleValue)
                    else:
                        X[i] = float(result.objectAtIndex_(i+1))
                if debug > 0:
                    print ('... Affinity ', A, 'J/mol')
                    print ('... X', X)
                    print ('... Convergence', result.objectAtIndex_(mu.size+1))
                    print ('... Iterations', result.objectAtIndex_(mu.size+2))
                    print ('... Affinity scalar', result.objectAtIndex_(mu.size+3))
                    print ('... Estimated error on affinity', result.objectAtIndex_(mu.size+4))
                return (A, X)
            else:
                return None
        else: # A module coded by hand or generated by coder (assume coder syntax)
            method = getattr(phs.module, phs.phase_obj+"affinity_and_composition", None)
            if method:
                if debug > 0:
                    print ('Calling tailored Affinity and Comp routine for',
                       phs.phase_obj)
                A, X, conv_flags = method(t, p, mu)
                if debug > 0:
                    print ('... Affinity ', A, 'J/mol')
                    print ('... X', X)
                    print ('... Convergence', conv_flags[0])
                    print ('... Iterations', conv_flags[1])
                    print ('... Affinity scalar', conv_flags[2])
                    print ('... Estimated error on affinity', conv_flags[3])
                return (A, X)
            else:
                return None

    def _affinity_and_comp_special_case(self, t, p, mu, debug=False,
                                        method='generic'):

        phs = self.phase
        return None


    def _update_mu_phi_ideal(self, t, p, mu):
        # set_m
        X_mask = self.X_mask
        X_min = self.X_min
        phs = self.phase
        # muM = self.muM
        # phi = self.phi

        muM = np.zeros(mu.size)
        phi = np.zeros(mu.size)

        # from IPython import embed; embed()
        X = np.full(mu.size, X_min)
        # np.copyto(X, np.full(mu.size, X_min))

        for ind in np.where(X_mask)[0]:
            X[ind] = 1.0

            # muM = mu0 + RTln X + RT ln gamma (for the endmember)
            # muM[ind] = phs.chem_potential(t, p, mol=X, endmember=ind)
            muM[ind] = phs.gibbs_energy(t, p, mol=X)
            X[ind] = X_min
            phi[ind] = mu[ind] - muM[ind]

        return muM, phi

    def _update_mu_phi(self, t, p, mu, X, chempot_update, deriv=False):
        # set_m

        X_mask = self.X_mask
        X_min = self.X_min
        phs = self.phase
        site_m = self.site_m
        # muM = self.muM
        # phi = self.phi

        # muM = np.zeros(mu.size)
        # phi = np.zeros(mu.size)
        # from IPython import embed; embed()

        # chempot = phs.chem_potential(t, p, mol=X)
        # chempot = chempot_update(t, p, mol=X)

        update = chempot_update(t, p, mol=X, deriv=deriv)
        muM = update[0]
        # muM = chempot_update(t, p, mol=X)

        # if deriv:
        #     dmudn = phs.gibbs_energy(t, p, mol=X, deriv={'dmol':2}).squeeze()
        #     # dmudn = phs.gibbs_energy(t, p, mol=X, deriv={'dmol':2})

        # np.copyto(muM, chempot)
        muM[~X_mask] = 0

        phi = mu-muM
        phi += 8.3143*t*site_m*np.log(X)
        phi[~X_mask] = 0

        if np.any(np.isnan(phi)):
            from IPython import embed; embed()

        # for ind in np.where(X_mask)[0]:
        #     # muM = mu0 + RTln X + RT ln gamma (for the endmember)
        #     muM[ind] = phs.chem_potential(t, p, mol=X, endmember=ind)
        #
        #     phi[ind] = mu[ind] - muM[ind]
        #
        #     # the phi now contains only the mu0 and RT ln gamma
        #     phi[ind] += 8.3143*t*site_m*np.log(X[ind])

        if deriv:
            dmudn = update[1]
            return muM, phi, dmudn
        else:
            return muM, phi

    def _update_X_A_direct(self, t, phi, mu):
        # replace mu=0 with X_mask
        X_mask = self.X_mask
        site_m = self.site_m

        RTm = 8.3143*t*site_m
        lnX_shft = phi/RTm
        lnX_max = np.max(lnX_shft[mu!=0])
        Xtot = np.sum(np.exp(lnX_shft[mu!=0]-lnX_max))
        nA_RTm = np.log(Xtot)+lnX_max
        lnX = lnX_shft-nA_RTm

        X_update = np.exp(lnX)
        # np.copyto(X,np.exp(lnX))
        X_update[~X_mask] = 0

        A = -RTm*nA_RTm
        return X_update, A

    def _update_mu_X_A(self, t, p, mu, X, chempot_update,
                       assume_ideal=False, deriv=False):

        if assume_ideal:
            update = self._update_mu_phi_ideal(t, p, mu)
        else:
            update = self._update_mu_phi(
                t, p, mu, X, chempot_update, deriv=deriv)

        muM, phi = update[0], update[1]

        if np.any(np.isnan(muM)) or np.any(np.isnan(phi)):
            print('mu-phi isnan')
            from IPython import embed; embed()

        # print('dG = ', np.dot(muM-mu, X))
        X_update, A = self._update_X_A_direct(t, phi, mu)

        if np.isnan(A) or np.any(np.isnan(X_update)):
            print('X-A isnan')
            from IPython import embed; embed()
        # print('A = ', A)
        # print('---')

        if deriv:
            dmudn = update[-1]
            return muM, X_update, A, dmudn

        else:
            return muM, X_update, A

    def _check_X_A_convergence(self, X, X_prev, A, A_prev, n_loop, debug):

        X_mask = self.X_mask
        ATOL = self.ATOL
        LOGXTOL = self.LOGXTOL

        converge_stats = {}

        logX_diff = np.log(X[X_mask]/X_prev[X_mask])
        # phi_diff = phi-phi_prev
        A_diff = A-A_prev
        logX_diff_max = np.max(np.abs(logX_diff))

        if debug:
            print('---')
            print('iter: {0:2d}'.format(n_loop),end=' ')
            print('Anew: {0:10.2f}'.format(A),end=' ')
            print('Aold: {0:10.2f}'.format(A_prev), end=' ')
            print('max(logX_diff) : {0:4.2f}'.format(logX_diff_max),
                                                     end=' ')
            print()
            # print('   X :', end=' ')
            # [print('{0:4.2e}'.format(iX), end=', ') for iX in X]
            # print()
            print('  logdX :', end=' ')
            [print('{0:+4.2e}'.format(idX), end=', ') for idX in logX_diff]
            # print()
            # print(' dphi :', end=' ')
            # [print('{0:+4.2e}'.format(idphi), end=', ') for idphi in phi_diff]
            print()

            # mol_lims = [1e-4,1]
            # plt.figure()
            # # plt.loglog(mol_endmem, X, 'bo')
            # plt.loglog(X_init, X, 'rx')
            # plt.plot(mol_lims, mol_lims, 'r--')
            # # plt.xlim(0,1)
            # # plt.ylim(0,1)
            # plt.xlim(mol_lims)
            # plt.ylim(mol_lims)
            # plt.pause(.01)

        A_converge = np.abs(A_diff) < ATOL
        X_converge=np.all(np.abs(logX_diff) < LOGXTOL)

        if A_converge and X_converge:
            converged = True
        else:
            converged = False

        converge_stats['converged'] = converged
        converge_stats['A'] = A
        converge_stats['X'] = X
        converge_stats['X_converge'] = X_converge
        converge_stats['A_converge'] = A_converge
        converge_stats['logX_diff'] = logX_diff
        converge_stats['logX_diff_max'] = logX_diff_max
        converge_stats['A_diff'] = A_diff
        converge_stats['n_loop'] = n_loop
        return  converge_stats


    # def chempot_update_log_quad(
    #                 t, p, mol=None, logX_ref=logX_ref,
    #                 mu_ref=mu_ref, dmudlogX=dmudlogX, X_mask=X_mask,
    #                 N=N, mu_full=mu_full):
    #
    #                 dlogX = np.log(mol[X_mask])-logX_ref
    #                 mu = mu_ref + np.dot(dmudlogX, dlogX)
    #                 mu_full[X_mask] = mu
    #                 return mu_full
    def affinity_and_comp(self, t, p, mu, X_init=None, site_m=None,
                          prev_wt=0.2, converge_method='direct',
                          full_output=False, dlogX_constr=0.1,
                          method='generic', save_hist=False,
                          debug=False, embed=False):
        """
        Given a temperature, pressure and chemical potential(s), compute and
        return a chemical affinity and phase composition.

        Parameters
        ----------
            Temperature (K) and pressure (bars)
        mu : ndarray
            Chemical potential(s) of endmember components in phase

            - For a stoichiometric phase, a 1-component 1-D numpy array

            - For a solution phase, a 1-D numpy array of length equal to the
              number of endmember components in the phase
        X_init : ndarray, optional
            Initial guess for composition (default None). If none is provided,
            then
        site_m : float, optional
            provides value of site-multiplicity for R*T*site_m terms.
            Speeds up repeated calculations. Can be determined from
            solution phase properties or manually using:
                site_m = self._est_site_mult(t, p)
        prev_wt : float, default 0.2
            Weighting factor for previous composition value,
            smoothing convergence results. Default value of 0.2 gives good
            results for most phases.
        converge_method : {'direct', 'approx', 'lstsq'}
            Determines convergence method for MEXQAL:
            'direct' - slow but safe, relying on analytic iterative updates
            'approx' - faster, using gibbs curvature matrix to find
                approximate solution within inner loop
            'lstsq' - fastest, uses gibbs curvature matrix and solves linear
                least squares problem to jump to best approximate solution
            In the end, convergence always verified w/ direct calculation.
        full_output : bool, default False
            If true, returns addition output dict with many intermiate
            and convergence-related values.
        debug : bool, def False
            Print iteration details
        method : str, def 'generic'
            Algorithm used:
            'generic' - generic algorithm on components,
            'species' - generic algorithm on species (> = components),
            'special' - algorthmic specifically coded in the phase instance,
                currently, only Objective-C coded methods are utilized,
                see _tailored_affinity_and_comp()

        Returns
        -------
        A, X : tuple
            A is a scalar equal to the chemical affinity of the phase relative
            to mu.

            X is a numpy array of mole fractions of endmember components in the
            phase. X is the same length as mu.

        Notes
        -----
        The algorithm used is from Ghiorso (2013) and is a globally convergent saturation
        state algorithm applicable to thermodynamic systems with a stable or
        metastable omni-component phase. Geochimica et Cosmochimica Acta, 103,
        295-300
        """

        phs = self.phase
        if method=='generic':
            assert mu.size==phs.endmember_num, 'chempot array provided to affinity_and_comp must have length equal to number of components in phase'
        elif method=='special':
            pass
        else: # method=species
            # should enforce species method...
            assert mu.size==self.Nc_full, 'chempot array provided to affinity_and_comp must have length equal to number of species in phase'


        muM = self.muM
        X = self.X
        X_prev = self.X_prev


        # Handle special case calculations
        if mu.size == 1:
            if mu[0] == 0.0:
                return (999999.0, np.array([1.0]))
            else:
                mu0 = phs.chem_potential(t, p)
                if debug:
                    print (mu[0], mu0)
                return (mu0-mu[0], np.array([1.0]))

        if method == 'special':
            result = self._tailored_affinity_and_comp(t, p, mu, debug=debug)
            if result is not None:
                return result

        # Initialize variables
        # mu=0 is a special value meaning ignore this component
        X_mask = np.full(mu.size, False)
        X_mask[np.nonzero(mu)[0]] = True
        self.X_mask = X_mask

        if debug:
            print ('Mole fraction mask: ', X_mask)

        assume_ideal = False
        if X_init is None:
            X[:] = self.X_min

            # from IPython import embed; embed()
            X[mu==0] = 0
            X_init = X.copy()
            assume_ideal = True
        else:
            X_init = np.array(X_init, dtype=float)
            X = X_init.copy()
            assert X.size==mu.size, (
                'If initial X value is provided, '
                'must match size of chempot array mu'
                )
            X[X<self.X_min] = self.X_min
            X[mu==0] = 0
            X /= X.sum()

        A_prev = sys.float_info.max
        if site_m is None:
            site_m = self._est_site_mult(t, p)
        self.site_m = site_m

        if debug:
            print ('Computed site multiplicity:', self.site_m)

        if embed:
            from IPython import embed; embed()

        X_hist = []
        mu_hist = []
        np.copyto(X_prev, X_init)

        chempot_update_exact = self.chempot_update

        muM, X_update, A = self._update_mu_X_A(
            t, p, mu, X, chempot_update_exact, assume_ideal=assume_ideal)

        if save_hist:
            X_hist.append(X.copy())
            mu_hist.append(muM.copy())

        X_avg = prev_wt*X_prev + (1-prev_wt)*X_update
        np.copyto(X, X_avg)

        if debug:
            print('pre-convergence X = ', X)
            print('muM pre-converge = ', muM)

        if converge_method=='direct':
            converge_stats, local_history = self._converge_X_A(
                t, p, mu, muM, A, X, X_prev, prev_wt,
                debug, chempot_update=chempot_update_exact)
            n_loop = converge_stats['n_loop']

            if save_hist:
                X_hist.extend(local_history['X'])
                mu_hist.extend(local_history['mu'])

        elif converge_method=='approx':
            np.copyto(X_prev, X)
            muM, X_update, A, dmudn = self._update_mu_X_A(
                t, p, mu, X, chempot_update_exact, deriv=True)

            X_ref = X.copy()
            mu_ref = muM.copy()
            if save_hist:
                X_hist.append(X_ref)
                mu_hist.append(muM.copy())

            iter_max = self.iter_max
            iter_max_inner = 30*iter_max

            debug_approx = False
            debug_approx = debug

            N=X_mask.size
            X = X_ref.copy()
            A_prev = A
            n_loop = 0
            while True:
                n_loop += 1
                A_prev = A
                np.copyto(X_prev, X)

                mu_ref_full = mu_ref.copy()
                logX_ref_full = np.log(X_ref)

                logX_ref = logX_ref_full[X_mask]
                mu_ref = mu_ref_full[X_mask]

                dmudlogX_full = dmudn*np.tile(X_ref, (X_ref.size,1))

                dmudlogX = dmudlogX_full[X_mask][:,X_mask]
                mu_full = np.zeros(N)

                def chempot_update_log_quad(
                    t, p, mol=None, deriv=False, logX_ref=logX_ref,
                    mu_ref=mu_ref, dmudlogX=dmudlogX, X_mask=X_mask,
                    N=N, mu_full=mu_full):

                    assert not deriv, 'deriv cannot be used in log_quad update'
                    dlogX = np.log(mol[X_mask])-logX_ref
                    mu = mu_ref + np.dot(dmudlogX, dlogX)
                    mu_full[X_mask] = mu
                    return mu_full,

                approx_converge_stats, local_history = self._converge_X_A(
                    t, p, mu, muM, A, X, X_prev, prev_wt,
                    debug_approx, chempot_update=chempot_update_log_quad,
                    iter_max=iter_max_inner)

                X_approx_update = local_history['X'][-1]
                X_ref = X_approx_update.copy()
                muM, X_update, A, dmudn = self._update_mu_X_A(
                    t, p, mu, X_approx_update, chempot_update_exact, deriv=True)

                mu_ref = muM.copy()

                if save_hist:
                    local_history['X'].append(X.copy())
                    local_history['mu'].append(muM.copy())

                A_prev = A  # THIS IS WRONG!!!

                np.copyto(X, X_update)
                # X_avg = prev_wt*X_prev + (1-prev_wt)*X_update

                converge_stats = self._check_X_A_convergence(
                    X, X_prev, A, A_prev, n_loop, debug)

                if converge_stats['converged'] or (n_loop >= iter_max):
                    break

        elif converge_method=='lstsq':
            dlogX_thresh = 1e-1
            np.copyto(X_prev, X)
            muM, X_update, A, dmudn = self._update_mu_X_A(
                t, p, mu, X, chempot_update_exact, deriv=True)

            X_ref = X.copy()
            mu_ref = muM.copy()
            if save_hist:
                X_hist.append(X_ref)
                mu_hist.append(muM.copy())

            iter_max = self.iter_max
            iter_max_inner = 30*iter_max

            debug_approx = False

            N=X_mask.size
            X = X_ref.copy()
            A_prev = A
            n_loop = 0
            while True:
                n_loop += 1
                A_prev = A
                np.copyto(X_prev, X)

                mu_ref_full = mu_ref.copy()
                logX_ref_full = np.log(X_ref)

                logX_ref = logX_ref_full[X_mask]
                mu_ref = mu_ref_full[X_mask]

                dmudlogX_full = dmudn*np.tile(X_ref, (X_ref.size,1))

                try:
                    dmudlogX = dmudlogX_full[X_mask][:,X_mask]
                except:
                    from IPython import embed;embed()
                mu_full = np.zeros(N)

                Np = logX_ref.size

                Amat_equil = np.hstack((dmudlogX, np.ones((Np,1))))
                Amat_Xnorm = np.hstack((X_ref[X_mask], 0))
                Amat_dlogX = np.hstack((np.eye(Np)/dlogX_constr,
                                        np.zeros((Np,1))))
                Amat = np.vstack((Amat_equil, Amat_Xnorm))
                # Amat = np.vstack((Amat_equil, Amat_Xnorm, Amat_dlogX))

                # print(Amat_equil.shape)
                # print(Amat_Xnorm.shape)
                # print(Amat.shape)

                b_equil = mu[X_mask]-mu_ref
                b_Xnorm = 0
                b_dlogX = np.zeros(Np)

                b = np.hstack((b_equil, b_Xnorm))
                # b = np.hstack((b_equil, b_Xnorm, b_dlogX))

                result = np.linalg.lstsq(Amat, b, rcond=None)
                xsoln = result[0]

                dlogX = xsoln[:-1]
                A = xsoln[-1]

                if debug:
                    print('dlogX = ', dlogX)
                    print('A = ', A)
                    print('dlogX direct = ', np.log(X_update/X_ref))

                # if np.any(np.abs(dlogX) > 3*dlogX_constr):
                if np.any(np.abs(dlogX) > dlogX_thresh):
                    X_approx_update = X_update.copy()
                    # X_approx_update = np.zeros(X_ref.size)
                    # X_approx_update[X_mask] = np.exp(dlogX)*X_ref[X_mask]
                    if debug:
                        print(np.max(np.abs(dlogX)))
                        print('*** use direct update ***')

                else:
                    X_approx_update = np.zeros(X_ref.size)
                    X_approx_update[X_mask] = np.exp(dlogX)*X_ref[X_mask]

                # X_approx_update = local_history['X'][-1]
                X_ref = X_approx_update.copy()
                muM, X_update, A, dmudn = self._update_mu_X_A(
                    t, p, mu, X_approx_update, chempot_update_exact, deriv=True)

                mu_ref = muM.copy()

                if save_hist:
                    local_history['X'].append(X.copy())
                    local_history['mu'].append(muM.copy())

                A_prev = A  # THIS IS WRONG!!!

                # np.copyto(X, X_update)
                np.copyto(X, X_approx_update)
                # X_avg = prev_wt*X_prev + (1-prev_wt)*X_update

                converge_stats = self._check_X_A_convergence(
                    X, X_prev, A, A_prev, n_loop, debug)

                if converge_stats['converged'] or (n_loop >= iter_max):
                    break


            # assert False, 'Not yet implemented.'

        else:
            assert False, converge_method+' is not a valid convergence method.'

        A = converge_stats['A']
        X = converge_stats['X']

        if full_output:
            output = {}
            output['iter_num'] = n_loop
            output['converged'] = converge_stats['converged']
            output['logdX'] = converge_stats['logX_diff']
            output['logdX_max'] = converge_stats['logX_diff_max']
            output['dA'] = A-A_prev
            output['X_hist'] = X_hist
            output['mu_hist'] = mu_hist
            return A, X, output

        else:
            return A, X


    def _converge_X_A(self, t, p, mu, muM, A, X, X_prev,
                      prev_wt, debug,
                      iter_max=None, chempot_update=None, n_loop=0):

        # One of these vars is changing rather than being copied to, it needs fixing

        # phi = self.phi
        # phi_prev = self.phi_prev
        # muM = self.muM
        # X = self.X
        # X_prev = self.X_prev
        if iter_max is None:
            iter_max = self.iter_max

        local_history = {}
        local_history['X'] = []
        local_history['mu'] = []


        converged = False
        while True:
            n_loop += 1
            A_prev = A
            # X_prev = X
            # phi_prev = phi
            np.copyto(X_prev, X)
            # np.copyto(phi_prev, phi)

            muM, X_update, A = self._update_mu_X_A(
                t, p, mu, X, chempot_update)

            if debug:
                print('---------')
                print('nloop ', n_loop)
                print('A = ', A)
                print('X_update = ', X_update)
                print('muM = ', muM)

            local_history['X'].append(X.copy())
            local_history['mu'].append(muM.copy())

            X_avg = prev_wt*X + (1-prev_wt)*X_update
            # Why does asignment and copyto give diff levels of convergence???
            # X = X_avg.copy()
            np.copyto(X, X_avg)

            converge_stats = self._check_X_A_convergence(
                X, X_prev, A, A_prev, n_loop, debug)

            if converge_stats['converged'] or (n_loop >= iter_max):
                break

        return converge_stats, local_history

    def determine_phase_stability(self, t, p, moles, debug=0, maxiter=250,
                                  threshold=-0.1,
                                  X_bnds=[0.01,.99], X_min_init=.05,
                                  X_min_endmem=1e-6, method='L-BFGS-B',
                                  return_all_results=False):
        """
        Calculate if the specified phase is stable with respect to unmixing.

        Returns:
        --------
        result: tuple
            bool, ndarray - unmixing detected; None or composition of second phase
        result: list of tuples
            [float, ndarray] - [function value, associated composition] if return_all_results
            is passed as True
        """
        # if phase_name == state.omni_phase():
        #     return False, None
        # phase = state.phase_d[phase_name]
        # moles = phase['moles']
        moles = np.array(moles)

        phs = self.phase

        if phs.endmember_num == 1:
            return False, None

        # assert moles.size == self.endmember_num, (
        #     'size of moles must match endmember number')

        tot_moles = np.sum(moles)

        if tot_moles <= 0:
            return False, None
        if debug > 0:
            print ('')
            print ('Unmixing calculation for', phs.phase_name, moles/tot_moles)

        mu0 = []
        nc = phs.endmember_num

        for i in range(0,nc):
            x = np.full(nc,X_min_endmem)
            x[i] = 1.0-np.sum(x)
            mu0.append(phs.gibbs_energy(t, p, mol=x))
        gHat = phs.gibbs_energy(t, p, mol=moles)
        dgdn = phs.gibbs_energy(t, p, mol=moles, deriv={'dmol':1})[0]
        for i in range(0,nc):
            gHat -= mu0[i]*moles[i]
            dgdn[i] -= mu0[i]
        gHat /= tot_moles

        def deltaG(x, *args):
            t, p, moles, gHat, dgdn, mu0 = args
            f = phs.gibbs_energy(t, p, mol=x)
            df = phs.gibbs_energy(t, p, mol=x, deriv={'dmol':1})[0]
            for i in range(0,nc):
                f -= mu0[i]*x[i]
                df[i] -= mu0[i]
            f /= np.sum(x)
            f -= gHat + np.matmul(x/np.sum(x)-moles/np.sum(moles), dgdn)
            df -= dgdn
            return f, df

        bound_set = []
        for i in range(0,nc):
            bound_set.append(X_bnds)

        lowest_fun = 0.0
        lowest_comp = None
        debug_flag = True if debug > 1 else False
        if return_all_results:
            result_l = []
        for i in range(0,nc):
            x = np.full(nc, X_min_init)
            x[i] = 1.0-np.sum(x)
            result = optim.minimize(deltaG, x,
                args=(t, p, moles, gHat, dgdn, mu0),
                method=method, jac=True,
                bounds=bound_set,
                options={'maxiter':250, 'disp':debug_flag})
            if debug > 0:
                print (result)
                print ('')
            if return_all_results:
                result_l.append((result.fun, result.x))
            if result.fun < lowest_fun:
                lowest_fun = result.fun
                lowest_comp = result.x
        if return_all_results:
            return result_l
        elif lowest_fun < threshold:
            return True, lowest_comp
        else:
            return False, None



    def affinity_and_comp_legacy(self, t, p, mu, debug=False, method='generic'):
        """
        Given a temperature, pressure and chemical potential(s), compute and
        return a chemical affinity and phase composition USING LEGACY VERSION.

        Parameters
        ----------
        t,p : float
            Temperature (K) and pressure (bars)
        mu : ndarray
            Chemical potential(s) of endmember components in phase

            - For a stoichiometric phase, a 1-component 1-D numpy array

            - For a solution phase, a 1-D numpy array of length equal to the
              number of endmember components in the phase
        debug : bool, def False
            Print iteration details
        method : str, def 'generic'
            Algorithm used:
            'generic' - generic algorithm on components,
            'species' - generic algorithm on species (> = components),
            'special' - algorthmic specifically coded in the phase instance,
                currently, only Objective-C coded methods are utilized,
                see _tailored_affinity_and_comp()
        Returns
        -------
        A, X : tuple
            A is a scalar equal to the chemical affinity of the phase relative
            to mu.

            X is a numpy array of mole fractions of endmember components in the
            phase. X is the same length as mu.

        Notes
        -----
        The algorithm used is from Ghiorso (2013) and is a globally convergent saturation
        state algorithm applicable to thermodynamic systems with a stable or
        metastable omni-component phase. Geochimica et Cosmochimica Acta, 103,
        295-300
        """

        phs = self.phase
        assert mu.size==phs.endmember_num, 'chempot array provided to affinity_and_comp must have length equal to number of components in phase'
        if mu.size == 1:
            if mu[0] == 0.0:
                return (999999.0, np.array([1.0]))
            else:
                mu0 = phs.chem_potential(t, p)
                if debug:
                    print (mu[0], mu0)
                return (mu0-mu[0], np.array([1.0]))
        else:
            if method == 'special':
                result = self._tailored_affinity_and_comp(t, p, mu, debug=debug)
                if result is not None:
                    return result


            X_mask_ind = np.nonzero(mu)[0]

            if debug:
                print ('Mole fraction mask ind: ', X_mask_ind)
                # from IPython import embed; embed()

            X = np.full(mu.size, 1.0e-8)
            muM = np.zeros(mu.size)
            phi = np.zeros(mu.size)
            A_previous = sys.float_info.max
            loop = True
            n_loop = 0
            site_m = self._est_site_mult(t, p)
            if debug:
                print ('Computed site multiplicity:', site_m)

            while loop:
                for ind in X_mask_ind:
                    if n_loop == 0:
                        X[ind] = 1.0
                    # muM = mu0 + RTln X + RT ln gamma (for the endmember)
                    muM[ind] = phs.chem_potential(t, p, mol=X, endmember=ind)
                    if n_loop == 0:
                        X[ind] = 1.0e-8
                    phi[ind] = mu[ind] - muM[ind]
                    if n_loop > 0:
                        # the phi now contains only the mu0 and RT ln gamma
                        phi[ind] += 8.3143*t*site_m*np.log(X[ind])
                for i in range(0,X.size):
                    X[i] = 0.0
                denom = 1.0
                prod = 1.0
                for i in range(0, X_mask_ind[:-1].size):
                    ind = X_mask_ind[i]
                    ind_next = X_mask_ind[i+1]
                    r = np.exp((phi[ind_next]-phi[ind])/(8.3143*t*site_m))
                    prod  *= r
                    denom += prod
                    X[ind_next] = prod
                ind0 = X_mask_ind[0]
                X[ind0] = 1.0/denom
                for i in range(1,X_mask_ind.size):
                    ind = X_mask_ind[i]
                    X[ind] *= X[ind0]
                A = -(phi[ind0] - 8.3143*t*site_m*np.log(X[ind0]))
                if debug:
                    print('iter: {0:2d}'.format(n_loop),end=' ')
                    print('Anew: {0:10.2f}'.format(A),end=' ')
                    if n_loop > 0:
                        print('Aold: {0:10.2f}'.format(A_previous), end=' ')
                    else:
                        print('Aold:           ', end=' ')
                    print ('X: ', X)
                loop = abs(A - A_previous) > 0.1
                n_loop += 1
                A_previous = A
                if n_loop > 50:
                    loop = False

            return (A, X)