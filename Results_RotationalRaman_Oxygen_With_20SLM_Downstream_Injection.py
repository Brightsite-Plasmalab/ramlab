# 15 mm

mm15_700W_T = [2932,2798,2412,1990,1396] #[x,x,a,x,x]
mm15_700W_percentage_O2 = [0.8238,0.9118,0.9196,1,0.9633] #2: 2644, 0.8679, 0.0356, 0.0965, 10.6664, 1.0149, 0.3409
mm15_700W_percentage_O = [0.0977,0.0578,0.0193,0,0]
mm15_700W_percentage_N2 = [0.0785,0.0305,0.0611,0.0,0.0367]
mm15_700W_k0 = [10.0471,10.5364,10.5978,10.5875,10.5817] #5: 1404, 0.9684, 0, 0.0316, 10.5063, 1.033, 0.3661
mm15_700W_k1 = [1.0151,1.0143,1.014,1.0133,1.0138]
mm15_700W_k2 = [0.3336,0.3578,0.3263,0.3551,0.414]

# 30 mm

mm30_700W_T = [1471,1513,1326,1233,1309] #[x,x,x,x,x]
mm30_700W_percentage_O2 = [0.8453,0.8756,0.8361,0.7273,0.7634]
mm30_700W_percentage_O = [0,0,0,0,0]
mm30_700W_percentage_N2 = [0.1457,0.1244,0.1639,0.2727,0.2366]
mm30_700W_k0 = [10.3377,10.3743,10.6572,10.6592,10.6115]
mm30_700W_k1 = [1.0147,1.0114,1.0116,1.0101,1.0109]
mm30_700W_k2 = [0.3572,0.3664,0.366,0.4019,0.374]

#40mm
mm40_700W_T = [1685,1496,1051,1022,1041]  #[x,x,x,x,x]
mm40_700W_percentage_O2 = [0.7769,0.7025,0.5688,0.5907,0.6266]
mm40_700W_percentage_O = [0,0,0,0,0]
mm40_700W_percentage_N2 = [0.2231,0.2975,0.4312,0.4093,0.3734]
mm40_700W_k0 = [10.2292,10.2092,9.7554, 9.6886,9.7783]
mm40_700W_k1 = [1.0146,1.0155,1.0146,1.0168,1.0134]
mm40_700W_k2 = [0.3584,0.353,0.3525,0.3344,0.3364]

# 50 mm
mm50_700W_T = [1205,1195,1230,1143,1226] #[x,x,x,x,x]
mm50_700W_percentage_O2 = [0.6527,0.6407,0.6839,0.6018,0.6481]
mm50_700W_percentage_O = [0.0,0,0,0,0]
mm50_700W_percentage_N2 = [0.3473,0.3593,0.3161,0.3982,0.3519]
mm50_700W_k0 = [9.0546,9.3911,9.5841,9.7437,9.4178]
mm50_700W_k1 = [1.016,1.0162,1.0147,1.0138,1.0133]
mm50_700W_k2 = [0.3716,0.3528,0.3451,0.3602,0.3249]

# 70 mm
mm70_700W_T = [1002,999,1028,1039,944] #[x,x,x,x,x]
mm70_700W_percentage_O2 = [0.6191,0.6207,0.6285,0.6305,0.6145]
mm70_700W_percentage_O = [0.0,0,0,0,0]
mm70_700W_percentage_N2 = [0.3809,0.3793,0.3715,0.3695,0.3855]
mm70_700W_k0 = [8.9876,9.0152,8.9987,9.0029, 9.0064]
mm70_700W_k1 = [1.0098,1.0101,1.0123,1.0152,1.0129]
mm70_700W_k2 = [0.3172,0.3345,0.3404,0.345,0.3336]

# 100 mm
mm100_700W_T = [958,952,949,979,1005] #[x,x,x,x,x]
mm100_700W_percentage_O2 = [0.6105,0.6131,0.6133,0.6158,0.6194]
mm100_700W_percentage_O = [0.0,0,0,0,0]
mm100_700W_percentage_N2 = [0.3895,0.3869,0.3867,0.3842,0.3806]
mm100_700W_k0 = [9.0024,8.9924,9.008,9.0053,9.0015]
mm100_700W_k1 = [1.0183,1.0162,1.0144,1.0134,1.013]
mm100_700W_k2 = [0.3634,0.3481,0.3309,0.3538,0.3793]

import matplotlib.pyplot as plt
import matplotlib
import wedme
import numpy as np

power700W_T = [mm15_700W_T, mm30_700W_T, mm40_700W_T,mm50_700W_T, mm70_700W_T, mm100_700W_T]
power700W_O = [mm15_700W_percentage_O,mm30_700W_percentage_O,mm40_700W_percentage_O,mm50_700W_percentage_O, mm70_700W_percentage_O, mm100_700W_percentage_O]
power700W_N2 = [mm15_700W_percentage_N2,mm30_700W_percentage_N2,mm40_700W_percentage_N2,mm50_700W_percentage_N2, mm70_700W_percentage_N2, mm100_700W_percentage_N2]

radial_list = [0,2,4,6,8]
distance_list = [15,30,40,50,70,100]
x, y = np.meshgrid(distance_list, radial_list)
z_700 = []
z_700_O = []
z_700_N2 = []

for i in range(len(y)):
    z_700.append(np.zeros(len(y[i])))

    z_700_O.append(np.zeros(len(y[i])))

    z_700_N2.append(np.zeros(len(y[i])))
    for j in range(len(y[i])):
        z_700[i][j] = power700W_T[j][i]

        z_700_O[i][j] = power700W_O[j][i]

        z_700_N2[i][j] = power700W_N2[j][i]

cmap_name = 'nipy_spectral'

from Results_RotationalRaman_Oxygen_With_5_SLM_Downstream_Injection import z_700 as z_700_5SLM
from Results_RotationalRaman_Oxygen_With_5_SLM_Downstream_Injection import z_700_O as z_700_O_5SLM
from Results_RotationalRaman_Oxygen_With_5_SLM_Downstream_Injection import z_700_N2 as z_700_N2_5SLM

z_700_N2_5SLM_scaled = list(map(lambda x: x /np.max(z_700_N2_5SLM), z_700_N2_5SLM))
z_700_N2_scaled = list(map(lambda x: x /np.max(z_700_N2), z_700_N2))




fig, ax= plt.subplots(4,2, sharex='all', layout="constrained")
fig.suptitle("5SLM                                                                                     20SLM",
             fontsize=32)
fig.set_size_inches(26,13)
im0_0 = ax[0,0].contourf(x,y,z_700_5SLM, vmin=500,levels=np.arange(500,3510,10), vmax=3510, cmap=cmap_name)
ax[0,0].scatter(x,y,color="grey")
ax[0,0].plot(40,0, marker='o', color='black', markersize=20)
im1_0 = ax[1,0].contourf(x,y,z_700_O_5SLM, vmin=0,levels=np.arange(0,0.3505,0.005), vmax=0.3505, cmap=cmap_name)
ax[1,0].scatter(x,y,color="grey")
ax[1,0].plot(40,0, marker='o', color='black', markersize=20)
im2_0 = ax[2,0].contourf(x,y,z_700_N2_5SLM, vmin=0,levels=np.arange(0,0.5005,0.005), vmax=0.5005, cmap=cmap_name)
ax[2,0].scatter(x,y,color="grey")
ax[2,0].plot(40,0, marker='o', color='black', markersize=20)
im3_1 = ax[3,0].contourf(x,y,z_700_N2_5SLM_scaled, vmin=0,levels=np.arange(0,1.01,0.01), vmax=1.01, cmap=cmap_name)
ax[3,0].scatter(x,y,color="grey")
ax[3,0].plot(40,0, marker='o', color='black', markersize=20)



im0_1 = ax[0,1].contourf(x,y,z_700, vmin=500,levels=np.arange(500,3510,10), vmax=3510, cmap=cmap_name)
ax[0,1].scatter(x,y,color="grey")
ax[0,1].plot(40,0, marker='o', color='black', markersize=20)
im1_1 = ax[1,1].contourf(x,y,z_700_O, vmin=0,levels=np.arange(0,0.3505,0.005), vmax=0.3505, cmap=cmap_name)
ax[1,1].scatter(x,y,color="grey")
ax[1,1].plot(40,0, marker='o', color='black', markersize=20)
im2_1 = ax[2,1].contourf(x,y,z_700_N2, vmin=0,levels=np.arange(0,0.5005,0.005), vmax=0.5005, cmap=cmap_name)
ax[2,1].scatter(x,y,color="grey")
ax[2,1].plot(40,0, marker='o', color='black', markersize=20)
im3_1 = ax[3,1].contourf(x,y,z_700_N2_scaled, vmin=0,levels=np.arange(0,1.01,0.01), vmax=1.01, cmap=cmap_name)
ax[3,1].scatter(x,y,color="grey")
ax[3,1].plot(40,0, marker='o', color='black', markersize=20)

plt.subplots_adjust(hspace=0.32)

ticks_T = np.arange(500,4100,600).tolist()
tick_O = np.arange(0,0.42,0.07).tolist()
tick_N2 = np.arange(0,0.6,0.1).tolist()
ticks_N2_scaled = np.arange(0,1.2,0.2).tolist()


fig.supylabel('Radial distance (mm)', fontsize=28)
fig.supxlabel('Downstream distance (mm)', fontsize=28)

ax[0,0].text(-0.1, 7.5, "(a)",
        size=28, font="serif", weight="bold")
ax[0,1].text(-0.1, 7.5, "(b)",
        size=28, font="serif", weight="bold")

cbar0 = fig.colorbar(im0_1, ax=ax[0,:], ticks=ticks_T, pad=0.01)
cbar1 = fig.colorbar(im1_1, ax=ax[1,:], ticks=tick_O, pad=0.01)
cbar2 = fig.colorbar(im2_1, ax=ax[2,:], ticks=tick_N2, pad=0.01)
cbar3 = fig.colorbar(im3_1, ax=ax[3,:], ticks=ticks_N2_scaled, pad=0.01)
cbar0.ax.set_title("T (K)", fontsize=24, pad=16)
cbar1.ax.set_title("O (%)", fontsize=24, pad=16)
cbar2.ax.set_title("N$_2$ (%)", fontsize=24, pad=16)
cbar3.ax.set_title("N$_2$ (scaled) (%)", fontsize=24, pad=16)
plt.savefig("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Analyzed\\Oxygen\\20_SLM_N2_vs_Power_Rainbow.jpg", dpi=300
            ,bbox_inches='tight')
plt.clf()