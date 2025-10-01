import matplotlib
import scipy.integrate as inte

mm0_T = [3838, 4061]
mm0_I_O1 = [3.5661]
mm0_I_O = [0.6495]
mm0W_wG = [0.9844, 1.0289]
mm0_wL = [1.031, 1.031]
mm0_k0 = [10.7944, 10.4601]
mm0_k1 = [1.0198, 1.02]
mm0_k2 = [0.5142549, 0.4330]
mm0_k3 = [0,4.99998]

mm0_0W_T = [296,296,296,296,296]
mm0_0W_ratio_I = [0,0,0,0,0]
mm0_0W_wG = [0.9497,0.9501,0.9517,0.952,0.9498]
mm0_0W_wL = [1.0309,1.0313,1.0324,1.0331,1.0313]
mm0_0W_k0 = [9.4282,9.3411,9.1625,8.711,9.4544]
mm0_0W_k1 = [1.022,1.0251,1.0244,1.027,1.0231]
mm0_0W_k2 = [1.08e-05,6.9e-06,4e-07,-5.79e-05,7.4e-06]

# 15 mm
mm15_400W_T = [2887,2628,1906,1087,585] #[x,x,x,x,x]
mm15_400W_O = [0.073,0.0269,0,0,0]
mm15_400W_wG = [0.9764,0.9734,0.9655,0.9558,0.9501]
mm15_400W_wL = [1.0311,1.0311,1.0311, 1.0311,1.031]
mm15_400W_k0 = [5.4968,5.9014,6.3564, 6.7192,6.7378]
mm15_400W_k1 = [1.0201,1.0201,1.02,1.0199,1.0201]
mm15_400W_k2 = [0.4145,0.4311,0.4207,0.4074,0.4196]
mm15_500W_T = [2898,2764,2430,1502,807] #[x,x,x,x,x]
mm15_500W_O = [0.0814,0.0445,0.0047,0,0]
mm15_500W_wG = [0.9771,0.9752,0.9716,0.9604,0,9528]
mm15_500W_wL = [1.0314,1.0312,1.0312,1.0315,1.031]
mm15_500W_k0 = [8.814,8.2253,8.2063,8.9048,8.9882]
mm15_500W_k1 = [1.02,1.0201,1.0199,1.0201, 1.0205]
mm15_500W_k2 = [0.3608,0.3941,0.4033,0.4238, 0.4273]
mm15_600W_T = [3195,3012,2786,1752,927] #[x,a,a,a,a]
mm15_600W_O = [0.1909,0.1081,0.0483,0,0]
mm15_600W_wG = [0.9804, 0.9778, 0.9752, 0.9633,0.954]
mm15_600W_wL = [1.032, 1.0308,1.031,1.031,1.0312]
mm15_600W_k0 = [9.3016, 8.914,9.0036,8.554,9.6504]
mm15_600W_k1 = [1.0201,1.0184,1.0186,1.0162,1.0245]
mm15_600W_k2 = [0.4095,0.4184,0.4172,0.5005,0.434]
mm15_700W_T = [3444,3209,2915,2043,1131] #[x,x,x,x,x]
mm15_700W_ratio_I = [0.5368,0.2442,0.0903,0.0,0]
mm15_700W_O = [0.3492,0.1963,0.0828,0.0,0]
mm15_700W_wG = [0.9825,0.9798,0.9765,0.9667,0.9564]
mm15_700W_wL = [1.0311, 1.0312, 1.0313,1.0311,1.0307]
mm15_700W_k0 = [8.8466,9.0367,9.627,9.2946,9.0276]
mm15_700W_k1 = [1.0163,1.0172,1.0204,1.0184,1.0172]
mm15_700W_k2 = [0.4398,0.4427,0.4066,0.4178,0.4263]

# 30 mm
mm30_400W_T = [2702,2466,1879,1127,575]  #[x,x,x,x,x]
mm30_400W_O = [0.0362,0.0131,0,0,0]
mm30_400W_wG = [0.9742,0.9716,0.9478,0.9562,0.9497]
mm30_400W_wL = [1.0315,1.031,1.0314, 1.031,1.0313]
mm30_400W_k0 = [5.9882,5.4084,5.5415, 5.7808,5.9056]
mm30_400W_k1 = [1.02,1.02,1.0199,1.0201,1.0199]
mm30_400W_k2 = [0.414,0.381,0.409,0.4615,0.4992]
mm30_500W_T = [2902,2508,1940,1176,628] #[0,0,0,0,0]
mm30_500W_O = [0.0821,0.0042,0,0,0]
mm30_500W_wG = [0.9766,0.9721,0.657,0.9565,0.9521]
mm30_500W_wL = [1.0311,1.0311,1.031,1.0306,1.0326]
mm30_500W_k0 = [6.6811,5.6609,5.5573,5.3753,6.0561]
mm30_500W_k1 = [1.0201,1.0201,1.02,1.0201, 1.018]
mm30_500W_k2 = [0.3728,0.4339,0.4359,0.4457, 0.2441]
mm30_600W_T = [2997,2816,2288,1441,669]  #[x,x,x,x,x]
mm30_600W_O = [0.1104,0.0558,0,0,0]
mm30_600W_wG = [0.9775, 0.9756, 0.9696, 0.9599,0.951]
mm30_600W_wL = [1.0311, 1.0311,1.0311,1.0311,1.0311]
mm30_600W_k0 = [6.406, 6.5103,6.5473,6.4976,6.2762]
mm30_600W_k1 = [1.0201,1.02,1.0201,1.0201,1.0201]
mm30_600W_k2 = [0.3922,0.3877,0.4055,0.4106,0.4435]
mm30_700W_T = [3154,3121,2610,1803,827] #[x,x,x,x,x]
mm30_700W_O = [0.1834,0.1484,0.0169,0,0]
mm30_700W_wG = [0.9796,0.9591,0.9731,0.9639,0.9528]
mm30_700W_wL = [1.0312, 0.9791, 1.031,1.0313,1.0311]
mm30_700W_k0 = [6.9505,5.5817,7.6965,7.9254,7.9653]
mm30_700W_k1 = [1.012,1.0204,1.0201,1.02,1.02]
mm30_700W_k2 = [0.403,0.4534,0.4291,0.415,0.4414]

# 50 mm
mm50_400W_T = [2371,2168,1686,1159,730]  #[x,x,x,x,x]
mm50_400W_O = [0.0056,0,0,0,0]
mm50_400W_wG = [0.9705,0.9684,0.9627,0.9568,0.9516]
mm50_400W_wL = [1.0311,1.0311,1.0311, 1.0312,1.0309]
mm50_400W_k0 = [6.5187,7.6558,7.5031,7.4971,7.5036]
mm50_400W_k1 = [1.0201,1.0202,1.0201,1.02,1.02]
mm50_400W_k2 = [0.4203,0.3941,0.4254,0.4248,0.3976]
mm50_500W_T = [2714,2508,1940,1176,628] #[x,x,x,x,x]
mm50_500W_O = [0.0422,0.0042,0,0,0]
mm50_500W_wG = [0.9746,0.9721,0.657,0.9565,0.9521]
mm50_500W_wL = [1.0311,1.0311,1.031,1.0306,1.0326]
mm50_500W_k0 = [5.6581,5.6609,5.5573,5.3753,6.0561]
mm50_500W_k1 = [1.0201,1.0201,1.02,1.0201, 1.018]
mm50_500W_k2 = [0.4168,0.4339,0.4359,0.4457, 0.2441]
mm50_600W_T = [2997,2829,2162,1380,757]  #[x,x,x,x,x]
mm50_600W_O = [0.1103,0.0543,0,0,0]
mm50_600W_wG = [0.9777, 0.9457, 0.968, 0.9592,0.952]
mm50_600W_wL = [1.0312, 1.0312,1.0314,1.0308,1.031]
mm50_600W_k0 = [3.9162, 4.3299,4.874,5.0846,5.1151]
mm50_600W_k1 = [1.0199,1.0201,1.0202,1.020,1.0199]
mm50_600W_k2 = [0.4371,0.4135,0.3849,0.4255,0.4201]
mm50_700W_T = [3161,2837,2432,1824,787] #[x,x,x,x,x]
mm50_700W_O = [0.1748,0.064,0.0025,0,0]
mm50_700W_wG = [0.9795,0.9758,0.9712,0.9642,0.9706]
mm50_700W_wL = [1.0312, 1.0311, 1.031,1.0309,0.9767]
mm50_700W_k0 = [6.7572,6.6395,6.3851,5.7163,5.5076]
mm50_700W_k1 = [1.02,1.02,1.02,1.020,1.0175]
mm50_700W_k2 = [0.4153,0.4251,0.4014,0.4029,0.4538]

# 70 mm
mm70_400W_T = [2165,2146,1670,1274,850]  #[x,x,x,x,x]
mm70_400W_O = [0,0,0,0,0]
mm70_400W_wG = [0.968,0.9679,0.9625,0.958,0.9352]
mm70_400W_wL = [1.0311,1.0309,1.0311, 1.0311,1.031]
mm70_400W_k0 = [5.8069,5.7833,5.7356, 5.7674,5.9291]
mm70_400W_k1 = [1.02,1.0202,1.02,1.02,1.0131]
mm70_400W_k2 = [0.4222,0.4689,0.4349,0.424,0.4268]
mm70_500W_T = [2657,2376,2080,1504,906] #[x,x,x,x,x]
mm70_500W_O = [0.0295,0.0055,0,0,0]
mm70_500W_wG = [0.974,0.9705,0.967,0.9608,0.9538]
mm70_500W_wL = [1.0312,1.0311,1.0309,1.0315,1.031]
mm70_500W_k0 = [6.2421,6.0596,5.7559,5.2183,4.8098]
mm70_500W_k1 = [1.0201,1.0201,1.0202,1.0201, 1.02]
mm70_500W_k2 = [0.4174,0.4178,0.4994,0.4794, 0.4333]
mm70_600W_T = [2796,2596,2269,1757,1114]  #[x,x,x,x,x]
mm70_600W_O = [0.0534,0.0218,0,0,0]
mm70_600W_wG = [0.9754, 0.9728, 0.9695, 0.9634,0.9562]
mm70_600W_wL = [1.0311, 1.0314,1.0311,1.0312,1.0304]
mm70_600W_k0 = [6.3946, 7.4368,6.9552,7.0388,7.2799]
mm70_600W_k1 = [1.02,1.02,1.02,1.02,1.0202]
mm70_600W_k2 = [0.4091,0.3824,0.3748,0.3954,0.4232]
mm70_700W_T = [2966,2782,2317,1827,1145] #[x,x,x,x,x]
mm70_700W_O = [0.0924,0.0557,0.0126,0,0]
mm70_700W_wG = [0.9773,0.9752,0.9703,0.9639,0.9569]
mm70_700W_wL = [1.0314, 1.0312, 1.031,1.0313,1.0312]
mm70_700W_k0 = [7.795,7.5479,7.59,7.4285,7.3722]
mm70_700W_k1 = [1.0201,1.0204,1.0202,1.0198,1.02]
mm70_700W_k2 = [0.3958,0.3826,0.3659,0.4817,0.4163]

# 100 mm
mm100_400W_T = [1863,1652,1369,1046,773] #[x,x,x,x,x]
mm100_400W_O = [0.0,0,0,0,0]
mm100_400W_wG = [0.9647,0.9623,0.9593,0.9551,0.9525]
mm100_400W_wL = [1.0311,1.0311,1.0314, 1.0315,1.0314]
mm100_400W_k0 = [6.544,6.6692,7.0324, 7.2921,7.4031]
mm100_400W_k1 = [1.02,1.0201,1.0199,1.0197,1.0199]
mm100_400W_k2 = [0.4249,0.4193,0.4256,0.3349,0.4029]
mm100_500W_T = [2363,2200,1761,1445,976] #[x,x,x,x,x]
mm100_500W_O = [0.0096,0,0,0,0]
mm100_500W_wG = [0.9704,0.969,0.9636,0.9605,0.9605]
mm100_500W_wL = [1.0312,1.0306,1.0313,1.0308,1.0338]
mm100_500W_k0 = [7.0179,7.1517,7.4007,7.3886,7.344]
mm100_500W_k1 = [1.0201,1.0197,1.0202,1.0201, 1.0195]
mm100_500W_k2 = [0.42,0.4032,0.4214,0.3851, 0.6431]
mm100_600W_T = [2648,2425,2133,1638,1016]  #[x,x,x,x,x]
mm100_600W_O = [0.027,0.0137,0,0,0]
mm100_600W_wG = [0.9735, 0.9712, 0.9676, 0.9615,0.9544]
mm100_600W_wL = [1.031, 1.0311,1.0316,1.0301,1.0307]
mm100_600W_k0 = [5.3297, 5.8644,6.0375,5.7192,5.8914]
mm100_600W_k1 = [1.02,1.02,1.0201,1.0181, 1.0207]
mm100_600W_k2 = [0.3925,0.4232,0.5335,0.2208,0.5738]
mm100_700W_T = [2906,2575,2314,1795,1190] #[x,x,x,x,x]
mm100_700W_O = [0.083,0.0211,0.0059,0,0]
mm100_700W_wG = [0.9764,0.9789,0.9702,0.9643,0.9566]
mm100_700W_wL = [1.0312, 1.0308, 1.0307,1.031,1.0315]
mm100_700W_k0 = [4.8848,4.7754,4.4457,4.1989,4.0651]
mm100_700W_k1 = [1.0201,1.0199,1.0198,1.02,1.0201]
mm100_700W_k2 = [0.4205,0.4097,0.4457,0.3765,0.4238]

radial_list = [0,2,4,6,8]
distance_list = [15,30,50,70,100]

power400W_T = [mm15_400W_T, mm30_400W_T, mm50_400W_T, mm70_400W_T, mm100_400W_T]
power500W_T = [mm15_500W_T, mm30_500W_T, mm50_500W_T, mm70_500W_T, mm100_500W_T]
power600W_T = [mm15_600W_T, mm30_600W_T, mm50_600W_T, mm70_600W_T, mm100_600W_T]
power700W_T = [mm15_700W_T, mm30_700W_T, mm50_700W_T, mm70_700W_T, mm100_700W_T]

power400W_O = [mm15_400W_O, mm30_400W_O, mm50_400W_O, mm70_400W_O, mm100_400W_O]
power500W_O = [mm15_500W_O, mm30_500W_O, mm50_500W_O, mm70_500W_O, mm100_500W_O]
power600W_O = [mm15_600W_O, mm30_600W_O, mm50_600W_O, mm70_600W_O, mm100_600W_O]
power700W_O = [mm15_700W_O, mm30_700W_O, mm50_700W_O, mm70_700W_O, mm100_700W_O]

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

x, y = np.meshgrid(distance_list, radial_list)
gas1 = ct.Solution('gri30.yaml')
temp = range(300,4000,10)
O2 = np.zeros(len(temp))
O = np.zeros(len(temp))
ratio = np.zeros(len(temp))

for i in range(len(temp)):
    temp_now = temp[i]
    gas1.TPX = temp_now, 101325, "O2:1"
    gas1.equilibrate("TP")
    O2[i] =(gas1["O2"].X[0])
    O[i] =(gas1["O"].X[0])
    ratio[i] = (gas1["O"].X[0]/(gas1["O2"].X[0]))

z_400 = []
z_500 = []
z_600 = []
z_700 = []

z_400_O = []
z_500_O = []
z_600_O = []
z_700_O = []

for i in range(len(y)):
    z_400.append(np.zeros(len(y[i])))
    z_500.append(np.zeros(len(y[i])))
    z_600.append(np.zeros(len(y[i])))
    z_700.append(np.zeros(len(y[i])))

    z_400_O.append(np.zeros(len(y[i])))
    z_500_O.append(np.zeros(len(y[i])))
    z_600_O.append(np.zeros(len(y[i])))
    z_700_O.append(np.zeros(len(y[i])))
    for j in range(len(y[i])):
        z_400[i][j] = power400W_T[j][i]
        z_500[i][j] = power500W_T[j][i]
        z_600[i][j] = power600W_T[j][i]
        z_700[i][j] = power700W_T[j][i]

        z_400_O[i][j] = power400W_O[j][i]
        z_500_O[i][j] = power500W_O[j][i]
        z_600_O[i][j] = power600W_O[j][i]
        z_700_O[i][j] = power700W_O[j][i]


temperature_list_temp = sum([[mm0_0W_T],power400W_T,power500W_T,power600W_T,power700W_T],[])
O_list_temp = sum([[mm0_0W_ratio_I],power400W_O,power500W_O,power600W_O,power700W_O],[])
temperature_list = sum(temperature_list_temp,[])
O_list = sum(O_list_temp,[])

plt.plot(radial_list,power400W_T[0])
plt.plot(radial_list, power400W_T[1])
plt.plot(radial_list, power400W_T[2])
plt.plot(radial_list, power400W_T[3])
plt.plot(radial_list, power400W_T[4])
plt.show()

integrals = [[],[],[],[],[]]
for i in range(len(integrals)):
    integrals[0].append(inte.simpson(power400W_T[i], x=radial_list))
    integrals[1].append(inte.simpson(power500W_T[i], x=radial_list))
    integrals[2].append(inte.simpson(power600W_T[i], x=radial_list))
    integrals[3].append(inte.simpson(power700W_T[i], x=radial_list))

matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['axes.labelsize'] =24
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['legend.fontsize'] = 20

fig, ax = plt.subplots(1,1)
fig.set_size_inches(20,10)
ax.plot(distance_list,integrals[0]/np.max(integrals[0]), label="400 W")
ax.plot(distance_list,integrals[1]/np.max(integrals[1]), label="500 W")
ax.plot(distance_list,integrals[2]/np.max(integrals[2]), label="600 W")
ax.plot(distance_list,integrals[3]/np.max(integrals[3]), label="700 W")
ax.set_xlabel('Distance (mm)')
ax.set_ylabel('Percentage of initial heat (-)')
ax.legend()
plt.savefig("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Analyzed\\Oxygen\\Heat_Loss_over_Distance.jpg", dpi=300
            ,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1,1)
fig.set_size_inches(20,10)
ax.plot(temp,O.tolist(),label="Chemical Equilibrium")
ax.scatter(temperature_list,O_list,color='red', label="Experimental Results")
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('O atom  Fraction (-)')
ax.legend()
plt.savefig("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Analyzed\\Oxygen\\Temperature_vs_O_Fraction.jpg", dpi=300
            ,bbox_inches='tight')
plt.show()


fig, ax= plt.subplots(4,1, sharex='all')
fig.set_size_inches(30,15)
im0 = ax[0].contourf(x,y,z_400, vmin=500,levels=np.arange(500,3450,100), vmax=3450)
ax[0].set_title('Power=400W')
im1 = ax[1].contourf(x,y,z_500, vmin=500,levels=np.arange(500,3450,100), vmax=3450)
ax[1].set_title('Power=500W')
im2 = ax[2].contourf(x,y,z_600, vmin=500,levels=np.arange(500,3450,100), vmax=3450)
ax[2].set_title('Power=600W')
im3 = ax[3].contourf(x,y,z_700, vmin=500,levels=np.arange(500,3450,100), vmax=3450)
ax[3].set_title('Power=700W')

fig.supylabel('Radial distance (mm)',x=0.1, fontsize=24)
ax[3].set_xlabel('Downstream distance (mm)')
cbar = fig.colorbar(im3, ax=ax)
plt.savefig("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Analyzed\\Oxygen\\No_Injection_T_vs_Power_contour100.jpg", dpi=300
            ,bbox_inches='tight')
plt.clf()

fig, ax= plt.subplots(4,1, sharex='all', sharey='all')
fig.set_size_inches(30,15)
im0 = ax[0].contourf(x,y,z_400, vmin=500,levels=np.arange(500,3450,10), vmax=3450)
ax[0].set_title('Power=400W')
im1 = ax[1].contourf(x,y,z_500, vmin=500,levels=np.arange(500,3450,10), vmax=3450)
ax[1].set_title('Power=500W')
im2 = ax[2].contourf(x,y,z_600, vmin=500,levels=np.arange(500,3450,10), vmax=3450)
ax[2].set_title('Power=600W')
im3 = ax[3].contourf(x,y,z_700, vmin=500,levels=np.arange(500,3450,10), vmax=3450)
ax[3].set_title('Power=700W')

fig.supylabel('Radial distance (mm)',x=0.1, fontsize=24)

ax[3].set_xlabel('Downstream distance (mm)')
cbar = fig.colorbar(im3, ax=ax)
plt.savefig("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Analyzed\\Oxygen\\No_Injection_T_vs_Power_contour10.jpg", dpi=300
            ,bbox_inches='tight')
plt.clf()

fig, ax= plt.subplots(4,1, sharex='all')
fig.set_size_inches(30,15)
im0 = ax[0].contourf(x,y,z_400_O, vmin=0,levels=np.arange(0,0.36,0.001), vmax=0.36)
ax[0].set_title('Power=400W')
im1 = ax[1].contourf(x,y,z_500_O, vmin=0,levels=np.arange(0,0.36,0.001), vmax=0.36)
ax[1].set_title('Power=500W')
im2 = ax[2].contourf(x,y,z_600_O, vmin=0,levels=np.arange(0,0.36,0.001), vmax=0.36)
ax[2].set_title('Power=600W')
im3 = ax[3].contourf(x,y,z_700_O, vmin=0,levels=np.arange(0,0.36,0.001), vmax=0.36)
ax[3].set_title('Power=700W')

fig.supylabel('Radial distance (mm)',x=0.1, fontsize=24)
ax[3].set_xlabel('Downstream distance (mm)')
cbar = fig.colorbar(im3, ax=ax)
plt.savefig("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Analyzed\\Oxygen\\No_Injection_O_vs_Power_contour100.jpg", dpi=300
            ,bbox_inches='tight')
plt.clf()
