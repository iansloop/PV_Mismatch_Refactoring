# -*- coding: utf-8 -*-

"""
This module contains the :class:`~pvmismatch.pvmismatch_lib.pvcell.PVcell`
object which is used by modules, strings and systems.
"""

from pvmismatch.pvmismatch_lib.pvconstants import PVconstants
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

# Defaults
RS = 0.004267236774264931  # [ohm] series resistance
RSH = 10.01226369025448  # [ohm] shunt resistance
ISAT1_T0 = 2.286188161253440E-11  # [A] diode one saturation current
ISAT2 = 1.117455042372326E-6  # [A] diode two saturation current
APH = 1.000426348582935  # photovoltaic current coefficient
ISC0_T0 = 6.3056  # [A] reference short circuit current
TCELL = 298.15  # [K] cell temperature
ARBD = 1.036748445065697E-4  # reverse breakdown coefficient
VRBD_ = -5.527260068445654  # [V] reverse breakdown voltage
NRBD = 3.284628553041425  # reverse breakdown exponent
CELLAREA = 153.33  # [cm^2] cell area
EG = 1.1  # [eV] band gap of cSi
ALPHA_ISC = 0.0003551  # [1/K] short circuit current temperature coefficient


class PVcell(object):
    """
    PVconstants - Class for PV constants
    :param Rs: series resistance [ohms]
    :param Rsh: shunt resistance [ohms]
    :param Isat1_T0: first saturation diode current at ref temp [A]
    :param Isat2: second saturation diode current [A]
    :param Isc0_T0: short circuit current at ref temp [A]
    :param cellArea: cell area [cm^2]
    :param aRBD: reverse breakdown coefficient
    :param VRBD: reverse breakdown voltage [V]
    :param nRBD: reverse breakdown exponent
    :param Eg: band gap [eV]
    :param alpha_Isc: short circuit current temp coeff [1/K]
    :param Tcell: cell temperature [K]
    :param Ee: incident effective irradiance [suns]
    """
    def __init__(self, Rs=RS, Rsh=RSH, Isat1_T0=ISAT1_T0, Isat2=ISAT2,
                 Isc0_T0=ISC0_T0, cellArea=CELLAREA, aRBD=ARBD, VRBD=VRBD_,
                 nRBD=NRBD, Eg=EG, alpha_Isc=ALPHA_ISC,
                 Tcell=TCELL, Ee=1., pvconst=PVconstants()):
        # user inputs
        self.Rs = Rs  #: [ohm] series resistance
        self.Rsh = Rsh  #: [ohm] shunt resistance
        self.Isat1_T0 = Isat1_T0  #: [A] diode one sat. current at T0
        self.Isat2 = Isat2  #: [A] diode two saturation current
        self.Isc0_T0 = Isc0_T0  #: [A] short circuit current at T0
        self.cellArea = cellArea  #: [cm^2] cell area
        self.aRBD = aRBD  #: reverse breakdown coefficient
        self.VRBD = VRBD  #: [V] reverse breakdown voltage
        self.nRBD = nRBD  #: reverse breakdown exponent
        self.Eg = Eg  #: [eV] band gap of cSi
        self.alpha_Isc = alpha_Isc  #: [1/K] short circuit temp. coeff.
        self.Tcell = Tcell  #: [K] cell temperature
        self.Ee = Ee  #: [suns] incident effective irradiance on cell
        self.pvconst = pvconst  #: configuration constants
        self.Icell = None  #: cell currents on IV curve [A]
        self.Vcell = None  #: cell voltages on IV curve [V]
        self.Pcell = None  #: cell power on IV curve [W]

    def __str__(self):
        fmt = '<PVcell(Ee=%g[suns], Tcell=%g[K], Isc=%g[A], Voc=%g[V])>'
        return fmt % (self.Ee, self.Tcell, self.Isc, self.Voc)

    def __repr__(self):
        return str(self)

    def __setattr__(self, key, value):
        if key not in ['pvconst', 'Icell', 'Vcell', 'Pcell']:
            value = np.float64(value)
        super(PVcell, self).__setattr__(key, value)
        # after all attributes have been initialized, recalculate IV curve
        # every time __setattr__() is called
        if hasattr(self, 'pvconst'):
            Icell, Vcell, Pcell = self.calcCell()
            super(PVcell, self).__setattr__('Icell', Icell)
            super(PVcell, self).__setattr__('Vcell', Vcell)
            super(PVcell, self).__setattr__('Pcell', Pcell)

    def update(self, **kwargs):
        """
        Update user-defined constants.
        """
        # TODO: use __dict__.update(), check for floats and update IV curve
        # self.__dict__.update(kwargs)
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @property
    def Vt(self):
        """
        Thermal voltage in volts.
        """
        return self.pvconst.k * self.Tcell / self.pvconst.q

    @property
    def Isc(self):
        return self.Ee * self.Isc0

    @property
    def Aph(self):
        """
        Photogenerated current coefficient, non-dimensional.
        """
        # short current (SC) conditions (Vcell = 0)
        Vdiode_sc = self.Isc * self.Rs  # diode voltage at SC
        Idiode1_sc = self.Isat1 * (np.exp(Vdiode_sc / self.Vt) - 1.)
        Idiode2_sc = self.Isat2 * (np.exp(Vdiode_sc / 2. / self.Vt) - 1.)
        Ishunt_sc = Vdiode_sc / self.Rsh  # diode voltage at SC
        # photogenerated current coefficient
        return 1. + (Idiode1_sc + Idiode2_sc + Ishunt_sc) / self.Isc

    @property
    def Isat1(self):
        """
        Diode one saturation current at Tcell in amps.
        """
        _Tstar = self.Tcell ** 3. / self.pvconst.T0 ** 3.  # scaled temperature
        _inv_delta_T = 1. / self.pvconst.T0 - 1. / self.Tcell  # [1/K]
        _expTstar = np.exp(
            self.Eg * self.pvconst.q / self.pvconst.k * _inv_delta_T
        )
        return self.Isat1_T0 * _Tstar * _expTstar  # [A] Isat1(Tcell)

    @property
    def Isc0(self):
        """
        Short circuit current at Tcell in amps.
        """
        _delta_T = self.Tcell - self.pvconst.T0  # [K] temperature difference
        return self.Isc0_T0 * (1. + self.alpha_Isc * _delta_T)  # [A] Isc0

    @property
    def Voc(self):
        """
        Estimate open circuit voltage of cells.
        Returns Voc : numpy.ndarray of float, estimated open circuit voltage
        """
        C = self.Aph * self.Isc + self.Isat1 + self.Isat2
        delta = self.Isat2 ** 2. + 4. * self.Isat1 * C
        return self.Vt * np.log(
            ((-self.Isat2 + np.sqrt(delta)) / 2. / self.Isat1) ** 2.
        )

    def calcCell(self):
        """
        Calculate cell I-V curves.
        Returns (Icell, Vcell, Pcell) : tuple of numpy.ndarray of float
        """
        Vdiode = self.Voc * self.pvconst.pts
        VPTS = self.VRBD * self.pvconst.negpts
        Vdiode = np.concatenate((VPTS, Vdiode), axis=0)
        Igen = self.Aph * self.Isc
        Idiode1 = self.Isat1 * (np.exp(Vdiode / self.Vt) - 1)
        Idiode2 = self.Isat2 * (np.exp(Vdiode / 2 / self.Vt) - 1)
        Ishunt = Vdiode / self.Rsh
        fRBD = np.asarray(1 - Vdiode / self.VRBD)
        fRBD[fRBD == 0] = np.finfo(np.float64).eps
        fRBD = self.aRBD * fRBD ** (-self.nRBD)
        Icell = Igen - Idiode1 - Idiode2 - Ishunt * (1 + fRBD)
        Vcell = Vdiode - Icell * self.Rs
        Pcell = Icell * Vcell
        return Icell, Vcell, Pcell

    # diode model
    #  *-->--*--->---*--Rs->-Icell--+
    #  ^     |       |              ^
    #  |     |       |              |
    # Igen  Idiode  Ishunt         Vcell
    #  |     |       |              |
    #  |     v       v              v
    #  *--<--*---<---*--<-----------=
    # http://en.wikipedia.org/wiki/Diode_modelling#Shockley_diode_model
    # http://en.wikipedia.org/wiki/Diode#Shockley_diode_equation
    # http://en.wikipedia.org/wiki/William_Shockley

    @staticmethod
    def f_Icell(Icell, Vcell, Isc, Aph, Rs, Vt, Isat1, Isat2, Rsh):
        """
        Objective function for Icell.
        :param Icell: cell current [A]
        :param Vcell: cell voltage [V]
        :param Isc: short circuit current at Tcell and Ee [A]
        :param Aph: photogenerated current coefficient
        :param Rs: series resistance [ohms]
        :param Vt: thermal voltage [V]
        :param Isat1: first diode saturation current at Tcell [A]
        :param Isat2: second diode saturation current [A]
        :param Rsh: shunt resistance [ohms]
        :return: residual = (Icell - Icell0) [A]
        """
        # arbitrary current condition
        Igen = Aph * Isc  # photogenerated current
        Vdiode = Vcell + Icell * Rs  # diode voltage
        Idiode1 = Isat1 * (np.exp(Vdiode / Vt) - 1.)  # diode current
        Idiode2 = Isat2 * (np.exp(Vdiode / 2. / Vt) - 1.)  # diode current
        Ishunt = Vdiode / Rsh  # shunt current
        return Igen - Idiode1 - Idiode2 - Ishunt - Icell

    def calcIcell(self, Vcell):
        """
        Calculate Icell as a function of Vcell.
        :param Vcell: cell voltage [V]
        :return: Icell
        """
        args = (np.float64(Vcell), self.Isc, self.Aph, self.Rs, self.Vt,
                self.Isat1, self.Isat2, self.Rsh)
        return fsolve(self.f_Icell, self.Isc, args)

    @staticmethod
    def f_Vcell(Vcell, Icell, Isc, Aph, Rs, Vt, Isat1, Isat2, Rsh):
        return PVcell.f_Icell(
            Icell, Vcell, Isc, Aph, Rs, Vt, Isat1, Isat2, Rsh
        )

    def calcVcell(self, Icell):
        """
        Calculate Vcell as a function of Icell.
        :param Icell: cell current [A]
        :return: Vcell
        """
        args = (np.float64(Icell), self.Isc, self.Aph, self.Rs, self.Vt,
                self.Isat1, self.Isat2, self.Rsh)
        return fsolve(self.f_Vcell, self.Voc, args)

    def plot(self):
        """
        Plot cell I-V curve.
        Returns cellPlot : matplotlib.pyplot figure
        """
        cell_plot = plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(self.Vcell, self.Icell)
        plt.title('Cell Reverse I-V Characteristics')
        plt.ylabel('Cell Current, I [A]')
        plt.xlim(self.VRBD - 1, 0)
        plt.ylim(0, self.Isc + 10)
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.plot(self.Vcell, self.Icell)
        plt.title('Cell Forward I-V Characteristics')
        plt.ylabel('Cell Current, I [A]')
        plt.xlim(0, self.Voc)
        plt.ylim(0, self.Isc + 1)
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.plot(self.Vcell, self.Pcell)
        plt.title('Cell Reverse P-V Characteristics')
        plt.xlabel('Cell Voltage, V [V]')
        plt.ylabel('Cell Power, P [W]')
        plt.xlim(self.VRBD - 1, 0)
        plt.ylim((self.Isc + 10) * (self.VRBD - 1), -1)
        plt.grid()
        plt.subplot(2, 2, 4)
        plt.plot(self.Vcell, self.Pcell)
        plt.title('Cell Forward P-V Characteristics')
        plt.xlabel('Cell Voltage, V [V]')
        plt.ylabel('Cell Power, P [W]')
        plt.xlim(0, self.Voc)
        plt.ylim(0, (self.Isc + 1) * self.Voc)
        plt.grid()
        return cell_plot