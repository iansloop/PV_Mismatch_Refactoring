from pvmismatch import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_mod_cell_ee(arr, landscape=True):
    from matplotlib import pyplot as plt
    if landscape:
        try:
            M = np.reshape(arr, (8, 12))[::-1]
            M[1::2, :] = M[1::2, ::-1]
        except:
            M = np.reshape(arr, (6, 11))[::-1]
            M[1::2, :] = M[1::2, ::-1]
    else:
        M = np.reshape(arr, (12, 8))[::-1]
        M[1::2, :] = M[1::2, ::-1]

    plt.matshow(M)
    plt.title('bit plot of cell illumination')


def plot_pvm_irr(pvm, landscape=True):
    from matplotlib import pyplot as plt
    P = [p.Ee for p in pvm.pvcells]
    plot_mod_cell_ee(P, landscape=True)


"""Inputs"""
# Represents how much shading the mesh screen provides (0 would let no light through)
mesh_shade = 0.427675

# Represents how much shading a cardboard sheet provides (about 1/8 inch thick)
# (Use 0.01 per Chetan's recommendation for PV Mismatch)
cardboard_shaded = 0.01  # 0.00299 #<-- experimental value

# Represents how much shading two plastic sheets provides (about 1/16 inch thick each)
plastic_shaded = 0.001616

# each column contains the indices of the cells within that column. Simplifies code for shading each column
module_columns_X = [tuple(range(0, 12)), tuple(range(12, 24)), tuple(range(24, 36)), tuple(range(36, 48)),
                    tuple(range(48, 60)),
                    tuple(range(60, 72)), tuple(range(72, 84)), tuple(range(84, 96))]
module_columns_A = [tuple(range(0, 11)), tuple(range(11, 22)), tuple(range(22, 33)), tuple(range(33, 44)),
                    tuple(range(44, 55)), tuple(range(55, 66))]

# index corresponds to phase of the shading test. Values are the fraction of the cell covered by the mesh
shaded_area_X = np.array([0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1])
shaded_area_A = np.array([0, 0.3951, 0.7902, 0.5805, 0.3708, 0.1611, 0.9513, 0.7416, 0.5318, 1])

# average irradiance before test starts
inital_irr = 0.71436

# average irradiance of west and east reference cell on roof (in suns, where 1 sun = 1000 W/m^2)
natural_irrad = np.array([inital_irr, 0.74796, 0.703228035, 0.752545, 0.75282, 0.751505, 0.746345, 0.742944995, 0.71332001, 0.7106])

# input experimental data for Pmp, Vmp, Imp
MPPT_X_exp = pd.read_excel(r'C:\Users\isloop\OneDrive for Business\Desktop\PVMismatch_Resources\NGT_XSERIES_POWER.xlsx')
MPPT_A_exp = pd.read_excel(r'C:\Users\isloop\OneDrive for Business\Desktop\PVMismatch_Resources\A_SERIES_STRING_POWER.xlsx')

# pull dc monitoring data from the east so that the power from the west can be normalized to it
East_DC_Power = pd.read_excel(
    r'C:\Users\isloop\OneDrive for Business\Desktop\PVMismatch_Resources\XSERIES_EAST_DC_MOD_POWER.xlsx')

# create series to serve as horizontal axis for 10 min averaged data
ten_min_interval = MPPT_X_exp['Timestamps'].iloc[20:120:10]
tmi = [ten_min_interval.iloc[j].tz_localize('UTC') for j in range(0,len(ten_min_interval))]
tmi = [tmi[k].tz_convert('America/Los_Angeles') for k in range(0,len(tmi))]
time_axis = [tmi[l].strftime('%H:%M') for l in range(0, len(tmi))]

# amount of irradiance received by the most recently shaded column
irrad_pattern_X = natural_irrad * (1 - shaded_area_X + shaded_area_X * mesh_shade)
irrad_pattern_A = natural_irrad * (1 - shaded_area_A + shaded_area_A * mesh_shade)

# average operating temperature of the cells during each phase
temp_array_X = np.array([31.679, 32.257, 33.582, 34.156, 34.569, 35.503, 36.933, 37.587, 37.546]) + np.array(
    [273.15, ] * 9)  # Kelvin
temp_array_A = np.array([31.947, 33.865, 35.373, 35.599, 36.262, 36.83, 37.882, 38.772, 37.897]) + np.array(
    [273.15, ] * 9)  # Kelvin

# temperature of the system before test starts
nominal_temp_X = 30.285 + 273.15  # kelvin
nominal_temp_A = 29.911 + 273.15  # kelvin

# irradiance received by a fully mesh-shaded cell
full_mesh_shade = mesh_shade * natural_irrad

""""Simulation"""
# X series (Gen E)
# X series cell and module properties
'''
RS=0.008#0.00477#0.0022904554199000655
RSH=250.01226369025448#14.866#5.524413919705285
ISAT1_T0=2.974132024e-12#5.615E-12#2.6951679883577537e-12
ISAT2_T0=2.394153128e-7#6.133E-7#9.078875806333005e-7
ISC0_T0=6.3056#6.39#6.590375
ARBD=1.036748445065697e-4#1.735E-2
BRBD=0#-0.6588
VRBD=-5.527260068445654#-4.50
NRBD=3.284628553041425#3.926
ALPHA_ISC=0.0003551#0.0003551
CELLAREA=153.33
EG=1.1#1.166
'''
'''
RS=0.00477#0.0022904554199000655
RSH=14.866#5.524413919705285
ISAT1_T0=5.615E-12#2.6951679883577537e-12
ISAT2_T0=6.133E-7#9.078875806333005e-7
ISC0_T0=6.39#6.590375
ARBD=1.735E-2
BRBD=-0.6588
VRBD=-4.50
NRBD=3.926
ALPHA_ISC=0.0003551
CELLAREA=153.33
EG=1.166
'''
# Datasheet
RS = 0.00477  # 0.0022904554199000655
RSH = 14.866  # 5.524413919705285
ISAT1_T0 = 5.615E-12  # 2.6951679883577537e-12
ISAT2_T0 = 6.133E-7  # 9.078875806333005e-7
ISC0_T0 = 6.66  # 6.590375
ARBD = 1.735E-2
BRBD = -0.6588
VRBD = -4.50
NRBD = 3.926
ALPHA_ISC = 0.0003551
CELLAREA = 153.33
EG = 1.166
'''
VRBD = np.float64(VRBD)
# Create PV system with inital irradiance and temperature
pvcell_X = pvcell.PVcell(Rs=RS, Rsh=RSH, Isat1_T0=ISAT1_T0, Isat2_T0=ISAT2_T0,
                         Isc0_T0=ISC0_T0, aRBD=ARBD, bRBD=BRBD, VRBD=VRBD,
                         nRBD=NRBD, Eg=EG, alpha_Isc=ALPHA_ISC)
pvmodule_X = pvmodule.PVmodule(pvcells=pvcell_X, Vbypass=[VRBD, VRBD, VRBD], cellArea=CELLAREA)
pvsys_X = pvsystem.PVsystem(numberStrs=2, numberMods=4, pvmods=pvmodule_X)
# pvsys_X = pvsystem.PVsystem(numberStrs=2,numberMods=4)
plt.ion()
pvsys_X.setSuns(inital_irr)
pvsys_X.setTemps(nominal_temp_X)

# shade first (index=0) and last (index=3) module completely on both X series strings (moving left to right)
# pvsys_X.setSuns({0: {0:cardboard_shaded*inital_irr, 3:cardboard_shaded*inital_irr}, 1:{0:cardboard_shaded*inital_irr, 3:cardboard_shaded*inital_irr}})
# per Chetan's recommendation, don't set suns lower than 0.01
pvsys_X.setSuns({0: {0: 0.01, 3: 0.01}, 1: {0: 0.01, 3: 0.01}})

# record initial Pmp, Imp, Vmp. First String is shaded, second string is unshaded control
MPPT_X = pd.DataFrame(
    columns=['Pmp', 'Pmp_control', 'Pmp_norm', 'Vmp', 'Vmp_control', 'Vmp_norm', 'Imp', 'Imp_control', 'Imp_norm'],
    index=range(0, 10))
MPPT_X.iloc[0] = {'Pmp': pvsys_X.pvmods[0][2].Pmod.max(), 'Vmp': pvsys_X.pvmods[0][2].Vmod.max(),
                  'Imp': pvsys_X.pvmods[0][2].Imod.max(),
                  'Pmp_control': pvsys_X.pvmods[1][2].Pmod.max(), 'Vmp_control': pvsys_X.pvmods[1][2].Vmod.max(),
                  'Imp_control': pvsys_X.pvmods[1][2].Imod.max()}

# shade cells, update temperatures, and record power for each phase of the test
for i in range(0, 9):
    # update the natural irradiance across ALL cells of 1st string, 2nd module
    pvsys_X.setSuns(
        {0: {1: {'cells': tuple(range(0, 96)),
                 'Ee': tuple([pvsys_X.pvmods[0][1].Ee[j][0] * natural_irrad[i+1] / natural_irrad[i] for j in
                              range(0, 96)])}}})

    #update natural irradiance on the unshaded control module, string 2, module 3
    pvsys_X.setSuns({1: {2: natural_irrad[i+1]}})
    #update cell temperatures
    pvsys_X.setTemps(temp_array_X[i])

    if i == 0:
        pvsys_X.setSuns({0: {1: {'cells': module_columns_X[0], 'Ee': (irrad_pattern_X[i+1],) * 12}}})
    else:
        # shading the rest of the columns
        pvsys_X.setSuns({0: {1: {'cells' : module_columns_X[i - 1], 'Ee' : (irrad_pattern_X[i+1],) * 12}}})

    #plot_pvm_irr(pvsys_X.pvmods[0][1])
    # shading half of first column

    MPPT_X.iloc[i + 1] = {'Pmp': pvsys_X.pvmods[0][1].Pmod.max(), 'Vmp': pvsys_X.pvmods[0][1].Vmod.max(),
                          'Imp': pvsys_X.pvmods[0][1].Imod.max(),
                          'Pmp_control': pvsys_X.pvmods[1][2].Pmod.max(),
                          'Vmp_control': pvsys_X.pvmods[1][2].Vmod.max(),
                          'Imp_control': pvsys_X.pvmods[1][2].Imod.max()}

MPPT_X['Pmp_norm'] = MPPT_X['Pmp'] / MPPT_X['Pmp_control']
MPPT_X['Vmp_norm'] = MPPT_X['Vmp'] / MPPT_X['Vmp_control']
MPPT_X['Imp_norm'] = MPPT_X['Imp'] / MPPT_X['Imp_control']

MPPT_X_exp['Type_D_norm'] = MPPT_X_exp['W5 Type D'] / East_DC_Power['Power E5 Type D']
MPPT_X_exp['Type_E_norm'] = MPPT_X_exp['W6 Type E'] / East_DC_Power['Power E6 Type E']

# plotting
# error_X = (MPPT_X['Pmp']-MPPT_X_experimental)/MPPT_X_experimental*100

# plot Pmp
plt.figure()
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Pmp_norm'], 'b*', label='PV Mismatch')
plt.plot(MPPT_X_exp['Timestamps'], MPPT_X_exp['Type_D_norm'], '--', label='Type D')
plt.plot(MPPT_X_exp['Timestamps'], MPPT_X_exp['Type_E_norm'], label='Type E')
# plt.plot(error_X,label='Error (%)')
plt.legend()
plt.title('X-Series Type E and D Module')
plt.xlabel('Test Interval')
plt.xticks(ten_min_interval, time_axis)
plt.ylabel('Normalized DC Power (Respectively)')
plt.yticks(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
plt.grid(True)
#
# # plot Imp
# plt.figure(2)
# plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Imp'], 'b*', label='PV Mismatch')
# plt.plot(MPPT_X_exp['Timestamps'],
#          MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W5'] + MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W6'],
#          label='Experimental')
# # plt.plot(error_X,label='Error (%)')
# plt.legend()
# plt.xlabel('Test Interval')
# plt.xticks(ten_min_interval)
# plt.ylabel('DC Imp (A)')
# plt.grid(True)
#
# # plot Vmp
# plt.figure(3)
# plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Vmp'], 'b*', label='PV Mismatch')
# plt.plot(MPPT_X_exp['Timestamps'], MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W5'], label='Experimental')
# # plt.plot(error_X,label='Error (%)')
# plt.legend()
# plt.xlabel('Test Interval')
# plt.xticks(ten_min_interval)
# plt.ylabel('DC Vmp (V)')
# plt.grid(True)

# pvsys_X.pvmods[0][1].plotMod()
# pvsys_X.plotSys()


    #A series
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

#create non-standard NGT cell and PV system with inital irradiance and temperature
pvcell_A = pvcell.PVcell(Rs=RS, Rsh=RSH, Isat1_T0=ISAT1_T0, Isat2_T0=ISAT2_T0,
                 Isc0_T0=ISC0_T0, aRBD=ARBD, VRBD=VRBD,
                 nRBD=NRBD, Eg=EG, alpha_Isc=ALPHA_ISC)
pvmodule_A = pvmodule.PVmodule(cell_pos=pvmodule.standard_cellpos_pat(11,[6]), Vbypass=VRBD, pvcells=pvcell_A, cellArea=cellArea)
pvsys_A = pvsystem.PVsystem(numberStrs=1,numberMods=4,pvmods=pvmodule_A)
pvsys_A_control = pvsystem.PVsystem(numberStrs=1,numberMods=4,pvmods=pvmodule_A)
plt.ion()
pvsys_A.setSuns(inital_irr)
pvsys_A_control.setSuns(inital_irr)
pvsys_A.setTemps(nominal_temp_A)
pvsys_A_control.setTemps(nominal_temp_A)

# shade the two modules at the ends of pvsys_A to simulate the plastic shading
#per Chetan's recommendation, don't set suns lower than 0.01
pvsys_A.setSuns({0: {0:inital_irr*plastic_shaded, 3:inital_irr*plastic_shaded}})

# record initial Pmp, Imp, Vmp (AC-String level). First String is shaded, second string is unshaded control
MPPT_A = pd.DataFrame(
    columns=['Pmp', 'Pmp_control', 'Pmp_norm', 'Vmp', 'Vmp_control', 'Vmp_norm', 'Imp', 'Imp_control', 'Imp_norm'],
    index=range(0, 10))
MPPT_A.iloc[0] = {'Pmp': sum([pvsys_A.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp': pvsys_A.Vmp,
                  'Imp': pvsys_A.Imp,
                  'Pmp_control': sum([pvsys_A_control.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp_control': pvsys_A_control.Vmp,
                  'Imp_control': pvsys_A_control.Imp}

#shade cells, update temperatures, and record power for each phase of the test
for i in range(0,9):
    # update irradiance on all cells in string 1, module 2 and 3 (mesh shaded modules)
    pvsys_A.setSuns(
        {0: {1: {'cells': tuple(range(0, 66)),
                 'Ee': tuple([pvsys_A.pvmods[0][1].Ee[j][0] * natural_irrad[i + 1] / natural_irrad[i] for j in
                              range(0, 66)])},
             2: {'cells': tuple(range(0, 66)),
                 'Ee': tuple([pvsys_A.pvmods[0][2].Ee[j][0] * natural_irrad[i + 1] / natural_irrad[i] for j in
                              range(0, 66)])}}})

    #update irradiance on string 1, modules 1 and 4 (plastic shaded modules), and string 2 (unshaded string)
    pvsys_A.setSuns({0: {0: natural_irrad[i+1]*plastic_shaded, 3: natural_irrad[i+1]*plastic_shaded}})
    pvsys_A_control.setSuns(natural_irrad[i+1])

    if i == 0 or i == 1:
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],)*11,module_columns_A[-1]],
                         2: [(irrad_pattern_A[i+1],)*11,module_columns_A[-1]]}})
    elif i == 4 or i == 5:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-3]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-3]]}})
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-4]],
                         2: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-4]]}})
    elif i == 6 or i == 7:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+2]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+2]]}})
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i+1]],
                         2: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i+1]]}})
    elif i == 8:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i + 2]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i + 2]]}})
    else:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+1]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+1]]}})
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i]],
                         2: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i]]}})
    pvsys_A.setTemps(temp_array_A[i])
    pvsys_A_control.setTemps(temp_array_A[i])

    MPPT_A.iloc[i + 1] = {'Pmp': sum([pvsys_A.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp': pvsys_A.Vmp,
                  'Imp': pvsys_A.Imp,
                  'Pmp_control': sum([pvsys_A_control.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp_control': pvsys_A_control.Vmp,
                  'Imp_control': pvsys_A_control.Imp}

    MPPT_A['Pmp_norm'] = MPPT_A['Pmp'] / MPPT_A['Pmp_control']
    MPPT_A['Vmp_norm'] = MPPT_A['Vmp'] / MPPT_A['Vmp_control']
    MPPT_A['Imp_norm'] = MPPT_A['Imp'] / MPPT_A['Imp_control']

    MPPT_A_exp['A_series_norm'] = MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_4'] / MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_3']
'''
# plotting
'''
# MPPT_A_experimental = np.array([506.465182,400.882196,332.6838467,304.707138,232.7229941,217.095214,209.072656,202.431444,188.462066,182.377072])
# error_A = (MPPT_A-MPPT_A_experimental)/MPPT_A_experimental*100
plt.figure()
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['Pmp_norm'], 'bo', label='PV Mismatch')
plt.plot(MPPT_A_exp['Timestamps'], MPPT_A_exp['A_series_norm'], label='A Series (NGT)')
# plt.plot(error_X,label='Error (%)')
plt.legend()
plt.title('4-Module String of NGT AC')
plt.xlabel('Test Interval')
plt.xticks(ten_min_interval, time_axis)
plt.ylabel('Normalized AC String Power (Respectively)')
plt.yticks(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
plt.grid(True)

plt.figure()
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['Pmp'], 'bo', label='PV Mismatch Pmp')
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['Pmp_control'], 'b*', label='PV Mismatch Pmp_control')
plt.plot(MPPT_A_exp['Timestamps'], MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_4']*1000, label='Pmp')
plt.plot(MPPT_A_exp['Timestamps'], MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_3']*1000, label='Pmp_control')
plt.legend()

plt.xlabel('Test Interval')
plt.xticks(ten_min_interval, time_axis)
plt.ylabel('AC String Power')
plt.grid(True)

sample_A = MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_4'].iloc[15:115:10]*1000
error_A = np.abs((np.array([j for j in sample_A]) - np.array([i for i in MPPT_A['Pmp']]))/np.array([j for j in sample_A]))*100

# pvsys_A.pvmods[0][3].plotMod()
# pvsys_A.pvmods[0][0].plotMod()
# pvsys_A.plotSys()

# IN PROGRESS
# experimental_power = np.array([0.4524,0.352,0.2649,0.2436,0.2234,0.2225,0.2163,0.2004,0.1925])*1000
# error = (MPPT_A-experimental_power)/experimental_power*100
