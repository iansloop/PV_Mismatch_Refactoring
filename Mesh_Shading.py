from pvmismatch import *
from matplotlib import pyplot as plt
import numpy as np

"""Inputs"""
#filler value for now. Represents how much shading the mesh screen provides (0 would let no light through)
mesh_shade= 0.2

#full shading approximation. setSuns doesn't like an exact 0
fully_shaded = 0.1

#each column contains the indicies of the cells within that column. Simplifies code for shading each column
module_columns_X = [tuple(range(0,12)),tuple(range(12,24)),tuple(range(24,36)),tuple(range(36,48)),tuple(range(48,60)),
                   tuple(range(60,72)),tuple(range(72,84)),tuple(range(84,96))]
module_columns_A = [tuple(range(0,11)),tuple(range(11,22)),tuple(range(22,33)),tuple(range(33,44)),tuple(range(44,55)),tuple(range(55,66))]

#index corresponds to phase of the shading test. Values are the fraction of the cell covered by the mesh
shaded_area_X = np.array([0.5,1,1,1,1,1,1,1,1])
shaded_area_A = np.array([0.3951,0.7902,0.5805,0.3708,0.1611,0.9513,0.7416,0.5318,1])

#irradiance from west reference cell
natural_irrad = np.array([0.74453,0.64889607,0.75002,0.75095,0.74982,0.74529,0.74247,0.71412001,0.71092])

#amount of irradiance received by the most recently shaded column
irrad_pattern_X = natural_irrad*(1 - shaded_area_X + shaded_area_X*mesh_shade)
irrad_pattern_A = natural_irrad*(1 - shaded_area_A + shaded_area_A*mesh_shade)

#average operating temperature of the cells during each phase
temp_array_X = np.array([31.679,32.257,33.582,34.156,34.569,35.503,36.933,37.587,37.546])  + np.array([273.15,]*9) #Kelvin
temp_array_A = np.array([31.947,33.865,35.373,35.599,36.262,36.83,37.882,38.772,37.897])  + np.array([273.15,]*9) #Kelvin

#temperature of the system before test starts
nominal_temp_X = 30.285+273.15 #kelvin
nominal_temp_A = 29.911+273.15 #kelvin

#average irradiance before test starts
inital_irr=0.71436

#A series (NGT) cell and module properties
ISC0_T0=10.2
RS=6.2e-3
RSH=38.7
ISAT1_T0=6.46e-12
ISAT2_T0=2.58e-7
ARBD=2.04
VRBD=-11.8211
NRBD=10
cellArea=258.3
EG = 1.166
ALPHA_ISC = 0.0003551



""""Simulation"""
    #X series
#Create PV system with inital irradiance
pvsys_X = pvsystem.PVsystem(numberStrs=2,numberMods=4)
plt.ion()
pvsys_X.setSuns(inital_irr)

#shade first (index=0) and last (index=3) module completely on both X series strings (moving left to right)
pvsys_X.setSuns({0: {0:fully_shaded, 3:fully_shaded}, 1:{0:fully_shaded, 3:fully_shaded}})

#assign nominal temperature and record initial power, and assign inital irradiance
power_array_X = []
pvsys_X.setTemps(nominal_temp_X)
power_array_X.append(pvsys_X.Pmp)

#shade cells, update temperatures, and record power for each phase of the test
for i in range(0,9):
    if i == 0: #shading half of first column
        pvsys_X.setSuns({0: {1: [(irrad_pattern_X[i],) * 12, module_columns_X[0]], 2: [(irrad_pattern_X[i],) * 12, module_columns_X[-1]]},
                         1: {1: [(irrad_pattern_X[i],) * 12, module_columns_X[0]], 2: [(irrad_pattern_X[i],) * 12, module_columns_X[-1]]}})
        pvsys_X.setTemps(temp_array_X[i])
        power_array_X.append(pvsys_X.Pmp)
    else:
        pvsys_X.setSuns({0: {1: [(irrad_pattern_X[i],)*12,module_columns_X[i-1]], 2: [(irrad_pattern_X[i],)*12,module_columns_X[-i]]},
                         1: {1: [(irrad_pattern_X[i],)*12,module_columns_X[i-1]], 2: [(irrad_pattern_X[i],)*12,module_columns_X[-i]]}})
        pvsys_X.setTemps(temp_array_X[i])
        power_array_X.append(pvsys_X.Pmp)

#plotting
power_array_X_experimental = np.array([506.465182,400.882196,332.6838467,304.707138,232.7229941,217.095214,209.072656,202.431444,188.462066,182.377072,417.951706])
plt.figure(1)
plt.plot(power_array_X,label='PV Mismatch')
plt.plot(power_array_X_experimental,label='Experimental')
plt.legend()
#pvsys_X.pvmods[0][1].plotMod()
#pvsys_X.plotSys()


    #A series
#create non-standard NGT cell and PV system with inital irradiance
pvcell_A = pvcell.PVcell(Rs=RS, Rsh=RSH, Isat1_T0=ISAT1_T0, Isat2_T0=ISAT2_T0,
                 Isc0_T0=ISC0_T0, aRBD=ARBD, VRBD=VRBD,
                 nRBD=NRBD, Eg=EG, alpha_Isc=ALPHA_ISC)
pvmodule_A = pvmodule.PVmodule(cell_pos=pvmodule.standard_cellpos_pat(11,[6]), pvcells=pvcell_A, cellArea=cellArea)
pvsys_A = pvsystem.PVsystem(numberStrs=2,numberMods=4,pvmods=pvmodule_A)
plt.ion()
pvsys_A.setSuns(inital_irr)

#shade last module in both strings completely
pvsys_A.setSuns({0: {3:fully_shaded}, 1: {3:fully_shaded}})

#assign nominal temperature and record initial power, and assign inital irradiance
power_array_A = []
pvsys_A.setTemps(nominal_temp_A)
power_array_A.append(pvsys_A.Pmp)

#irradiance received by a fully mesh-shaded cell
full_mesh_shade = mesh_shade*natural_irrad

#shade cells, update temperatures, and record power for each phase of the test
for i in range(0,9):
    if i == 0 or i == 1:
        pvsys_A.setSuns({0: {2: [(irrad_pattern_A[i],)*11,module_columns_A[-1]]},
                         1: {2: [(irrad_pattern_A[i],)*11,module_columns_A[-1]]}})
        pvsys_A.setTemps(temp_array_A[i])
        power_array_A.append(pvsys_A.Pmp)
    elif i == 4 or i == 5:
        pvsys_A.setSuns({0: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-3]]},
                         1: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-3]]}})
        pvsys_A.setSuns({0: {2: [(irrad_pattern_A[i],) * 11, module_columns_A[-4]]},
                         1: {2: [(irrad_pattern_A[i],) * 11, module_columns_A[-4]]}})
        pvsys_A.setTemps(temp_array_A[i])
        power_array_A.append(pvsys_A.Pmp)
    elif i == 6 or i == 7:
        pvsys_A.setSuns({0: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-i+2]]},
                         1: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-i+2]]}})
        pvsys_A.setSuns({0: {2: [(irrad_pattern_A[i],) * 11, module_columns_A[-i+1]]},
                         1: {2: [(irrad_pattern_A[i],) * 11, module_columns_A[-i+1]]}})
        pvsys_A.setTemps(temp_array_A[i])
        power_array_A.append(pvsys_A.Pmp)
    elif i == 8:
        pvsys_A.setSuns({0: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-i + 2]]},
                         1: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-i + 2]]}})
        pvsys_A.setTemps(temp_array_A[i])
        power_array_A.append(pvsys_A.Pmp)
    else:
        pvsys_A.setSuns({0: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-i+1]]},
                         1: {2: [(full_mesh_shade[i],) * 11, module_columns_A[-i+1]]}})
        pvsys_A.setSuns({0: {2: [(irrad_pattern_A[i],) * 11, module_columns_A[-i]]},
                         1: {2: [(irrad_pattern_A[i],) * 11, module_columns_A[-i]]}})
        pvsys_A.setTemps(temp_array_A[i])
        power_array_A.append(pvsys_A.Pmp)

#plotting
#pvsys_A.pvmods[0][3].plotMod()
#pvsys_A.pvmods[0][0].plotMod()
#pvsys_A.plotSys()

#IN PROGRESS
#experimental_power = np.array([0.4524,0.352,0.2649,0.2436,0.2234,0.2225,0.2163,0.2004,0.1925])*1000
#error = (power_array_A-experimental_power)/experimental_power*100
