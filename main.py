#导入matplotlib.pyplot模块
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
import pandas as pd
import scipy
import cmath
import progressbar

from scipy.integrate import quad

# import constants #导入物理常数

# from utils import *
from vidal import *


def main():
    # PDShape = KaiShape(omega_7s,  delta_wd, omega1, omega2,gamma_gb, delta_wp)
    # print(PDShape)
    # print(NDelta_kp(80000,Hg()))
    # Kaidraw(list)
    # DeltakpDraw(list)
   
    CdConDict = {"atomicN" : 114,
                 "confocal" : 0.001, # confocal parameter
        "T" :798.15, ##(*Temperature in Kelvin*)
        "P" : constants.Patom/17, ##pa
        "d" : 302*10**-12,#(*Diameter of atoms*)
        "L" : 0.5,##m
        "A" : 0.25*10**-6}     ##m^2



    #参考1956isotope shifts文献
    Cd114Isotope = pd.read_csv("./data/CdIsotope114.csv")


    # FinalStateList = [59219.734,53310.101,51483.98,59485.768,59497.868,62563.435,65134.783,65353.372,65358.881]
    # 列表数字只保留前面整数，"./data/Cd{}.csv"作为文件名列表
    # FinalStateFileList =  ["./data/Cd{}.csv".format(int(ele)) for ele in FinalStateList]


    #创建双y轴图表
    # 创建一个包含两个y轴的图表
    fig, ax1 = plt.subplots()

    # 创建第一个y轴
    ax1.set_xlabel('wavlen(nm)')
    ax1.set_ylabel('148.7nm Output Power(nW)')
    # 创建第二个y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('compensate angle')
    # 画图
    TestWavnum = 53310.101
    # MediatestateFilepath = "./data/gleb/Cd{}_Yu_gleb.csv".format(int(TestWavnum))
    MediatestateFilepath = "./data/gleb/Cd{}_gleb.csv".format(int(TestWavnum))
    # MediatestateFilepath = "./data/2024.1/Cd{}.csv".format(int(TestWavnum))
    # MediatestateFilepath = "./new/Cd{}.csv".format(int(TestWavnum))

    gamma_a=1
    gamma_b=wavenum2roundfreq(TestWavnum)
    gamma_c=0

    CdTestPool = VidalPool("Cd", TestWavnum, MediatestateFilepath, CdConDict, gamma_a, gamma_b, gamma_c)
    print("Cd{}pool N 为{}".format(TestWavnum,CdTestPool.N))
    print("一般多普勒展宽对应频率Ghz,具体 为{}".format(wavenum2freq(0.067)))
    print("148.7nm对应wavnum是{}".format(1/148.7*10**7))
    print("121.567nm对应wavnum是{}".format(wavlen2wavnum(121.567)))

    CdTestExpSynsDict = {"omega0" :wavenum2roundfreq(TestWavnum),#某一同位素的超参，omega0是到b态的跃迁频率, 都是圆频率
    "omega1" : wavenum2roundfreq(TestWavnum)/2,
    "omega2" : wavenum2roundfreq(TestWavnum)/2,
    "gamma_gb" : 2*constants.Pi *5*10**7 + 0.274*1e9,
    "omega_7s" : wavenum2roundfreq(TestWavnum)}



    # 计算高斯单位制下的ea0
    # print((4.80320427*10**-10* 5.291770859*10**-9 )**4/6/((h*10**7*c*10**2)**3))
    # print(4.80320427*10**-10* 5.291770859*10**-9/ea0)

    # AllShapeDraw(Cd53310ExpSynsDict, Cd53310Pool, Cd114Isotope,plot=False,test=True)

    # kai_effect = NLSusceptibility(wavlen2wavnum(257), HgPool2) * NLSusceptibility(wavlen2wavnum(121.57), HgPool2) * 8
    # p4 = FWMout(P1=2, P2=2, P3=1, wavnum1 = 31964.060, wavnum4 = 76466.925, ExpSynsDict = HgExpSynsDict,type = "Two Photon Resonance", Pool = HgPool)
    
    delta_wp = PressureBroadening(CdTestPool)
    print("delta_wp is {}".format(delta_wp))
    ShapeFactor = MultiIsotopeS( CdTestExpSynsDict["omega_7s"], CdTestExpSynsDict["gamma_gb"], delta_wp,Cd114Isotope,CdTestExpSynsDict,CdTestPool)
    
    # p4tight,p4tightperfect,bestN = FWMout(P1=0.58, P2=0.5, P3=1.2, wavnum1 = wavlen2wavnum(375.16), wavnum4 = wavlen2wavnum(148.35), ExpSynsDict = CdTestExpSynsDict,Isotope = Cd114Isotope, \
    #                                         type = "Three", Pool = CdTestPool,log = True,test = False,ShapeFactor= ShapeFactor)
    
    # print("p4tight is {}".format(p4tight))

    wavnum_arr,wavlen_arr = XAxisGenerator(lmin = 148.4,lmax = 148.3,datanumber=400)
    phis_array = []
    ele_Gbk_arr = []

    # 创建进度条
    widgets = [
        progressbar.FormatLabel(
            'Calculating Cd FWM output, whose b state is {}...'.format(TestWavnum),
        ),
        ' ',
        progressbar.Bar(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Timer(),
        ' ',]


    bar = progressbar.ProgressBar(maxval=len(wavnum_arr), widgets=widgets)
    bar.start()
    for i,ele in enumerate(wavnum_arr):
        #进度条更新
        bar.update(i+1)
        phis = Vidal_cal_in_gauss_unit(P1 = 0.58, P2= 0.5, P3=1.2, omega1=wavlen2wavnum(375.16), omega2=wavlen2wavnum(375.16), omega3=wavlen2wavnum(714), omega4=wavlen2wavnum(148.3), FWMPool = CdTestPool)
        # import ipdb;ipdb.set_trace()
        phis_array.append(phis)
        # print(p4tightperfect)
        # y2 = Delta_k(wavenum2roundfreq(ele),Cd59219Pool) #这里使用Deltak是没有考虑原子密度N的影响
        # fsquarefactor = abs(FShapeSquare(Cd59219Pool.confocal,y2)) 

        # f_array.append(fsquarefactor)
        # #f_array中等于0的点标红
        # if fsquarefactor == 0:
        #     ele_Gbk_arr.append(wavnum2wavlen(ele))


    bar.finish()
    #标签绘图
    # ax1.plot(wavlen_arr,p4parallelactual_array,label = "actual of " + str(TestWavnum))
    ax1.plot(wavlen_arr,phis_array,label = "phis of " + str(TestWavnum))
    #绿色的是补偿角度
    # ax2.plot(wavlen_arr, compensateangle_array,label = "compensate angle of " + str(TestWavnum),color = "green")

    ResonanceLineDraw(wavlen_arr,MediatestateFilepath)
        #标题 ，2005年的Hg 63928 Power Out
    plt.title("Cd Power Out".format(TestWavnum))
    plt.scatter(ele_Gbk_arr,np.zeros(len(ele_Gbk_arr)),c = "r")
    # plt.xlabel("wavelength(nm)")
    # plt.ylabel("Power(nW)")
    #纵坐标范围(0到12000)
    # ax1.set_ylim(0,24000)
    ax1.legend(loc='upper left')
    plt.savefig("Cd Power Out of {}.png".format(TestWavnum))



if __name__ == '__main__':
    main()




    







    

