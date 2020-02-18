from pvmismatch import *
from matplotlib import pyplot as plt
import numpy as np

#create string and module system (default 96c module)
pvsys = pvsystem.PVsystem(numberStrs=2,numberMods=4)
plt.ion()

pvsys.setSuns(0.8)
#f = pvsys.plotSys()

#shade bottom row of first string, first module
pvsys.setSuns({0: {0: [(0.2,)*8,(11,12,35,36,59,60,83,84)]}})
#f_shade = pvsys.plotSys()
pvsys.pvmods[0][0].plotMod() #plots first string, first module iv curve
pvsys.pvmods[0][0].plotCell() #plots all the cells (? I dont understand this one ?)


np.interp(pvsys.Vmp, pvsys.pvstrs[0].Vstring,pvsys.pvstrs[0].Istring) #calculate first string's Imp
pvsys.setTemps(50.+273.15) #Normal cell temp [K]
pvsys.setTemps({0:{0:[(100.+273.15,)*8,(11,12,35,36,59,60,83,84)]}})

#f_hot = pvsys.plotSys()

[Icell,Vcell,Pcell] = pvsys.pvmods[0][0].pvcells[0].calcCell()
I_op = pvsys.Imp/2 # I operating = operating current of the system
P_op = np.interp(I_op,Icell[1,:],Pcell[1,:])
