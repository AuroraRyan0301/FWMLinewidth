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
from tqdm import tqdm

from scipy.integrate import quad

import constants #导入物理常数





def freq2wavenum(freq):
    return freq/(constants.c*100)

def wavenum2freq(wn):
    return wn*constants.c*(10**2)

def roundfreq2wavenum(roundfreq):
    return roundfreq/(constants.c*100*2*constants.Pi)

def wavenum2roundfreq(wn):
    return wn*constants.c*(10**2)*2*constants.Pi #波数单位cm-1

def wavlen2roundfreq(wavlen):
    return constants.c*(10**9)/(wavlen*2*constants.Pi) #波长单位nm

def wavlen2wavnum(wavlen): #波长单位nm
    return 10**7 / wavlen

def wavnum2wavlen(wavnum):
    return 10**7 / wavnum

def mu2f(miu,omega): #将跃迁矩阵元转换为f值，miu输入应该为国际单位制，所以一般来说是1.5*ea0的形式，数量级很小，下同。omega为圆频率
    f = 2 * constants.me * omega/(3*constants.e**2 * constants.hbar) * miu**2
    return f

def f2mu(f, omega): #将f转换为跃迁矩阵元
    miu = ((f *3 * constants.e**2 * constants.hbar)/(2 * constants.me * omega))**0.5
    return miu

def csv_mu2f(csvfile):
    # 读取Hgwrong.csv,将gf的值用mu2f转换一下，然后再新建一列“gf_f”写入Hgwrong.csv
    Hgwrong = pd.read_csv(csvfile)
    Hgwrong["gf_f"] = Hgwrong.apply(lambda x: mu2f(x["gf"]*constants.ea0,x["f_energy_Hz"]*2*constants.Pi),axis=1)
    Hgwrong["fb_f"] = Hgwrong.apply(lambda x: mu2f(x["fb"]*constants.ea0,x["f_energy_Hz"]*2*constants.Pi),axis=1)
    Hgwrong.to_csv(csvfile,index=False)
    print(Hgwrong)

def csv_f2mu(csvfile):
    # 读取Hgwrong.csv,将gf_f的值用f2mu转换一下，然后写入Hgwrong.csv的gf列，fb列,注意输出单位是cm1，有乘ea0
    Hgwrong = pd.read_csv(csvfile)
    Hgwrong["gf"] = Hgwrong.apply(lambda x: f2mu(x["gf_f"],x["f_energy_Hz"]*2*constants.Pi)/constants.ea0,axis=1)
    Hgwrong["fb_f"] = Hgwrong["fb_f"].astype(float)
    #逐行读取，转换，写入
    f_sign = 1 #1代表正号，-1代表负号
    for i in range(len(Hgwrong)):
        #如果fb_f和上一行符号不一样，那么就取反,默认负一行符号为正号
        if i==0 and Hgwrong.loc[i,"fb_f"]<0:
            f_sign = f_sign * (-1)
        if i>0 and Hgwrong.loc[i,"fb_f"]*Hgwrong.loc[i-1,"fb_f"]<0:
            f_sign = f_sign * (-1)
        Hgwrong.loc[i,"fb"] = f2mu(abs(Hgwrong.loc[i,"fb_f"]),Hgwrong.loc[i,"f_energy_Hz"]*2*constants.Pi)/constants.ea0 * f_sign
    Hgwrong.to_csv(csvfile,index=False)
    print(Hgwrong)

def csv_wavnum2freq(csvfile):
    # 读取Hgwrong.csv,将wavnum的值用wavnum2freq转换一下，然后写入Hgwrong.csv的f_energy_Hz列
    Hgwrong = pd.read_csv(csvfile)
    Hgwrong["f_energy_Hz"] = Hgwrong.apply(lambda x: wavenum2freq(x["f_energy_wavnum"]),axis=1)
    Hgwrong.to_csv(csvfile,index=False)
    print(Hgwrong)

class FWMpool():
    def __init__(self,elename,EndEnergyWavNum,FstateFilepath,eleCondition):
        self.name = elename
        self.confocal = eleCondition["confocal"] #confocal parameter
        self.g = 0 # 0e:0 1S0 startL
        self.b = EndEnergyWavNum  
        self.L = eleCondition["L"] #Pool length in meter
        self.A = eleCondition["A"] #Pool area in meter square
        #immediate state
                # Define each channel as a dictionary
        #读取elename+'.csv'文件，创建channels属性，根据列名初始化为字典，每一行对应一个字典，每一列对应字典的键值，形成字典的列表self.channels
        self.FstateFilepath = FstateFilepath
        df = pd.read_csv(self.FstateFilepath)
        self.channels = df.to_dict('records')
    
        # print(self.channels)

        self.a = self.channels[0] # 读取字典第一个是对应w1,w2的channel, 3P1的能级，符号 2o:0
        
        self.atomicN = eleCondition["atomicN"] #质子和中子数
        self.T = eleCondition["T"] ##(*Temperature in Kelvin*)
        self.d = eleCondition["d"] #(*Diameter of atoms*) in meter
        self.P = eleCondition["P"] #pa
        # self.N = 8.84*10**23 # 平均分子数密度,2005年论文密度
        self.N = (self.P*constants.Avo)/(constants.R*self.T) # 平均分子数密度 
        self.m = constants.mp*self.atomicN 

    def ele2csv(self):
    #将FWM类中的channels属性导出为excel文件, 一共5列
        df = pd.DataFrame(self.channels)
        df.to_csv(self.name+'.csv',index=False,sep=',')
        print(df)

    def csv2ele(self):
    #读取csv文件，创建channels属性，根据列名初始化为字典，每一行对应一个字典，每一列对应字典的键值，形成字典的列表self.channels
        df = pd.read_csv(self.FstateFilepath)
        self.channels = df.to_dict('records')
        print(self.channels)


def NLSusceptibility(wavnum, ele: FWMpool): #注意这里omega应该是wavnum
    wavnum_prime = ele.b - wavnum
    Kai = 0
    for i in range(len(ele.channels)):
        Kai += ele.channels[i]['gf']*ele.channels[i]['fb']/(ele.channels[i]['f_energy_wavnum'] - wavnum) + \
        ele.channels[i]['gf']*ele.channels[i]['fb']/(ele.channels[i]['f_energy_wavnum'] - wavnum_prime)
    #如果虚部为正返回abs
    return Kai

#定义Plasma Dispersion Func
def KaiShape(omega_gb, gamma_doppler, omega1, omega2,gamma_gb,gamma_pressure):
    w = gamma_doppler /(2*((math.log(2))**0.5)) # 根据多普勒展宽计算w，每个能级对应w不同，因为Doppler broadening 違う
    gamma_homo_gb = gamma_gb + gamma_pressure #Lorentzian homogeneous linewidth of the transition,
    zeta = (omega1 + omega2 + gamma_homo_gb/2 * 1j- omega_gb)/w
    w_wavnum =  roundfreq2wavenum(w) #转换为cm-1的单位，之前的doppler展宽是圆频率，所以w也是圆频率，要转换为波数
    return PlasmaDispersion(zeta)/ w_wavnum

def PlasmaDispersion(a):
    # 写出plasma-dispersion function的积分

    def PDintegrand(x, a):
        return 1/(constants.Pi**0.5)*np.exp(-x**2)/(x-a)
    def real_func(x,a):
        return np.real(PDintegrand(x,a))
    def imag_func(x,a):
        return np.imag(PDintegrand(x,a))
    real_integral = quad(real_func, -np.inf, np.inf, args=(a,))
    imag_integral = quad(imag_func, -np.inf, np.inf, args=(a,))
    return real_integral[0]+imag_integral[0]*1j

#def const_initial():
    global mp, k, c, R, Avo, Patom, Pi, e,epsion0,me,a0,ea0, hbar, h
    mp = 1.6726*10**-27 #(*Proton mass*)
    k = 1.380649*10**-23 #(*Boltzmann constant*)

    c = 299792458.0 #(*Speed of light*) 一定要加.0，表示这个是浮点数
    R = 8.3145 #(*Ideal gas constant*)
    Avo = 6.02214*10**23#(*Avogadro constant*)
    Patom = 101325.0 #(*Atmospheric pressure*,Pa)

    Pi = 3.1415926535
    hbar = 6.62607015 * 10 **-34 / (2*Pi) #J*s
    h = 6.62607015 * 10 **-34 #J*s

    #声明电子电荷
    e = 1.602176634*10**(-19) #C
    a0 = 5.2917721067*10**-11            #bohr radius,m
    ea0 = e*a0

    epsion0 = 8.854187817 * 10**-12 #F/m
    me = 9.10938356 * 10**-31 #kg

def DopplerBroadening(omega0,Pool:FWMpool): #这个函数输入的omega0为圆频率也可以，返回的圆频率

    sigma = ((constants.k*Pool.T)/(Pool.m*constants.c**2)*omega0)**0.5 #这个地方的m是什么

    omegaD = (((8*constants.k*Pool.T*math.log(2))/(Pool.m*constants.c**2))**0.5 )* omega0

    return omegaD

def PressureBroadening(Pool:FWMpool):
    omegaP = Pool.N*Pool.d**2*((16 * constants.Pi*constants.k*Pool.T)/Pool.m)**0.5 * 2 * constants.Pi

    return omegaP

def k_NL(omega,element):
    return RefractiveIndex(omega,element)*omega/constants.c #k=wn/c

def NDelta_kp(omega,element:FWMpool): #这个函数输入的omega为波数
    # print("计算过程N为{}".format(element.N))
    # print("计算过程k为{}".format(k_NL(omega,element)+k_NL((element.b-omega),element)-2*k_NL(element.b/2,element)))
    # print("计算过程k1为{}".format(k_NL(omega,element)))
    # print("计算过程k2为{}".format(k_NL(element.b-omega,element)))
    # print("计算过程k3为{}".format(k_NL(element.b/2,element)))
    return  (k_NL(omega,element)+k_NL((wavenum2roundfreq(element.b)-omega),element)-2*k_NL(wavenum2roundfreq(element.b/2),element))/(element.N) 
#上面的公式1987原论文没有错，但少了后面两项会导致画的图不对（有个偏置），这里后面两项加不加只影响画图不影响最后deltak的计算（因为有个k1-k4抵消了）

def Delta_k(omega,element:FWMpool):
    return  (k_NL(omega,element)+k_NL((wavenum2roundfreq(element.b)-omega),element)-2*k_NL(wavenum2roundfreq(element.b/2),element)) #不考虑原子密度的影响

def RefractiveIndexSumPart(omega, Element:FWMpool):
    sum = 0
    omega_b = wavenum2roundfreq(Element.b)
    end_round_freq = omega_b
    for i,channel in enumerate(Element.channels):
        gf_subchan_energy = wavenum2roundfreq(channel["f_energy_wavnum"]) #中间态能量
        fb_subchan_energy = end_round_freq-gf_subchan_energy 
        sum += channel['gf_f']/(gf_subchan_energy**2-omega**2) #这里的omega是圆频率

        #没做if判断前，我总觉得这个地方应该再加fbchannel，但这个地方不能加，否则会出现多的共振峰，来源是另外一半channel，比如50000，另外一半是10000，会和70000的channel共振
        #做了if判断后正常了

        # if omega >0 and omega < omega_b: # 如果是正频率，不考虑负频率的channel，这里做了一定程度近似，避免出现多的共振峰
        #     sum += channel['gf_f']/(gf_subchan_energy**2-omega**2)
        #     # print("channel为{},此次sum为P{}".format(channel["f_energy_wavnum"],channel['gf']/(gf_subchan_energy**2-omega**2)))
        # else: 
        #     # sum += channel['gf_f']/(gf_subchan_energy**2-omega**2) 
        #     sum += channel['gf_f']/(gf_subchan_energy**2-omega**2) + channel['fb_f']/(fb_subchan_energy**2-omega**2) 
        # channel['fb']/(fb_subchan_energy**2-omega**2) #*ea0是为了将单位转换为国际单位制
        # if omega == 66274:
        #     print(channel["channel"])
        #     print(gf_subchan_energy)
        #     print(fb_subchan_energy)
        #     print(omega_b-omega)
        #     print(channel['gf']/(gf_subchan_energy**2-omega**2))
            # print(channel['fb']/(fb_subchan_energy**2-omega**2))
    # if omega == 66274:
    #     print(sum/((c*(10**2))**2))
    #     quit()
    # print("sum是{},wavnum是{}".format(sum,omega))
    return sum 
def RefractiveIndex(omega,Ele:FWMpool):
    n = 1+Ele.N*constants.e**2/(2*constants.epsion0*constants.me)*(RefractiveIndexSumPart(omega, Ele))
    return n

def GaussianShapeSquare(b,deltak):
    bdeltak = b*deltak
    if bdeltak > 0:
        return 0
    else:
        return constants.Pi**2 * (bdeltak)**4 * math.exp(bdeltak)
def FShapeSquare(b,deltak):
    bdeltak = b*deltak
    if bdeltak > 0:
        return 0
    else:
        return constants.Pi**2 * (bdeltak)**2 * math.exp(bdeltak)
def Kaidraw1987(elelist):
    #获取elelist长度，创建ele_Kai_arr空二维数组
    ele_Kai_arr = []
    omega_arr = np.arange(30000, 75000, 1)
    for ele in elelist:
    #创建波数数组，范围为30000到75000，步长为100
        Kai_arr = []
        for omega in omega_arr:
            y = NLSusceptibility(omega, ele)
            Kai_arr.append(y*(10**4))
        ele_Kai_arr.append(Kai_arr)

    #y取对数，x为横坐标画对数图
    # plt.yscale('log')
    # plt.plot(omega_arr, Hg_Kai_arr)
    # 设置y轴尺度-20到20
    plt.ylim(-20, 20)
    # y轴单位
    plt.ylabel('Kai * 10^4([ea0]^2/cm^-1)')
    plt.xlabel('wavenumber(cm^-1)')

    # Cd 和 Hg 数据画在同一张图里
    for i in range(len(elelist)):
        plt.plot(omega_arr, ele_Kai_arr[i], label=elelist[i].name)
    #加图例，左上角
    plt.legend(loc='upper left')
    # 加网格
    plt.grid(True)
    # 加标题
    plt.title('Kai Hg(1987)')
    plt.show()

def MultiIsotopeS(omega,gamma_gb,delta_wp,isotopeinfo,ExpSynsDict,Pool): #多同位素S(w1+w2)因子
    datas = 0
    #从datta中取出同位素信息,每行读取遍历
    for i in range(len(isotopeinfo)):
        omega_7s = isotopeinfo["omega"][i] + ExpSynsDict["omega0"]#到b态的圆频率
        omegaD = DopplerBroadening(omega_7s,Pool) #到b态的doppler 展宽
        y_iso = KaiShape(omega_7s,  omegaD, omega, 0,gamma_gb, delta_wp)
        datas += (abs(y_iso)**2) * isotopeinfo["abundance"][i]**2 #注意这个地方加权算法的特殊性
    return datas**0.5 #返回s，不是s的平方

def MultiShapeSquareGenerator(ExpSynsDict, Pool, isotopeinfo,plot,test = False): #test为True时，数据照搬2005的数据
    omegaD = DopplerBroadening(ExpSynsDict["omega0"],Pool) #到b态的doppler 展宽
    omegap = PressureBroadening(Pool) #到b态的气体压力展宽
    omega_7s, gamma_gb = ExpSynsDict["omega_7s"], ExpSynsDict["gamma_gb"]

    if test == True:
        omegaD = 2*constants.Pi * 2.15 *10**9
        omegap = 2*constants.Pi * 1.25 *10**9
        gamma_gb = 2*constants.Pi * 5 *10**6
    print("多普勒展宽为{}Hz".format(omegaD/2/constants.Pi))
    print("气体压力展宽为{}Hz".format(omegap/2/constants.Pi))
    print("自然展宽为{}Hz".format(gamma_gb/2/constants.Pi))

    delta_wd = omegaD
    delta_wp = omegap

    #创建波数数组，范围为40000到60000，步长为100
    S_arr = []
    center_value = roundfreq2wavenum(ExpSynsDict["omega0"])  # 中心值
    #center_value = roundfreq2wavenum(ExpSynsDict["omega0"])  # 中心值
    print("中心波长为{}".format(center_value))
    length = 1000  # 数列长度,或者说数据点个数
    bandwidth = 0.5

    # 生成中心值为某个波数的数列
    sequence = [center_value + i/length*2*bandwidth - bandwidth for i in range(length)]

    omega_arr = np.array(sequence)
    rfreq_arr = []
    #换算成圆频率数组
    for i,ele in enumerate(omega_arr):
        rfreq_arr.append(wavenum2roundfreq(ele))
    for rfreq in rfreq_arr:
            datas = MultiIsotopeS(rfreq, gamma_gb, delta_wp, isotopeinfo, ExpSynsDict, Pool)**2
            S_arr.append(datas)


    # 找出最大值，打印对应圆频率
    max_value = max(S_arr)
    max_index = S_arr.index(max_value)
    max_rfreq = rfreq_arr[max_index]
    print("最大值为{}，对应圆频率为{}".format(max_value, max_rfreq))
    print("最大值为{}，对应波数为{}".format(max_value, (roundfreq2wavenum(max_rfreq)-63928.120)))
    if plot == True:
        plt.plot(omega_arr, S_arr)
        plt.show()
    elif plot == False:
        pass
    return omega_arr, S_arr

def SingleShapeSquareGenerator(ExpSynsDict, Pool,plot=True,test=False): #test为True时，数据照搬2005的数据

    omegaD = DopplerBroadening(ExpSynsDict["omega0"],Pool) #到b态的doppler 展宽
    omegap = PressureBroadening(Pool)
    omega_7s, gamma_gb = ExpSynsDict["omega_7s"], ExpSynsDict["gamma_gb"]

    if test == True:
        omegaD = 2*constants.Pi * 2.15 *10**9
        omegap = 2*constants.Pi * 1.25 *10**9
        gamma_gb = 2*constants.Pi * 5 *10**6
    print("多普勒展宽为{}Hz".format(omegaD/2/constants.Pi))
    print("气体压力展宽为{}Hz".format(omegap/2/constants.Pi))
    print("自然展宽为{}Hz".format(gamma_gb/2/constants.Pi))

    delta_wd = omegaD
    delta_wp = omegap

    #创建波数数组，范围为40000到60000，步长为100
    S_arr = []
    center_value = roundfreq2wavenum(ExpSynsDict["omega0"])  # 中心值
    #center_value = roundfreq2wavenum(ExpSynsDict["omega0"])  # 中心值
    print("中心波长为{}".format(center_value))
    length = 1000  # 数列长度,或者说数据点个数
    bandwidth = 0.5

    # 生成中心值为某个波数的数列
    sequence = [center_value + i/length*2*bandwidth - bandwidth for i in range(length)]

    omega_arr = np.array(sequence)
    freq_arr = []
    #换算成频率数组
    for i,ele in enumerate(omega_arr):
        freq_arr.append(wavenum2roundfreq(ele))
    print(freq_arr[-3:-1])
    print(omega_7s)
    for freq in freq_arr:
        y = KaiShape(omega_7s,  delta_wd, freq, 0,gamma_gb, delta_wp)
        S_arr.append(abs(y)**2)
    if plot == True:
        plt.plot(omega_arr, S_arr)
        plt.show()
    elif plot == False:
        pass
    return omega_arr, S_arr

def DeltakpDataPointDraw(wavnum, ele):
    return NDelta_kp(wavnum,ele)* (10**4)#Δ_kp = k(w)+k(w_b-w)/N，这个地方* (10**4)是换算到cm^2

def DeltakpDraw(elelist):
    ele_Deltakp_arr = []
    wavnum_arr = np.arange(30000, 75000, 0.1)
    for ele in elelist:
    #创建波数数组，范围为40000到60000，步长为100
        Deltakp_arr = []

        for wavnum in  wavnum_arr:
            y = NDelta_kp(wavenum2roundfreq(wavnum),ele)
            y = y* (10**4) #Δ_kp = k(w)+k(w_b-w)/N，这个地方* (10**4)是换算到cm^2
            Deltakp_arr.append(y*10**16)
        ele_Deltakp_arr.append(Deltakp_arr)
    plt.ylim(-1, 1)

    # Cd 和 Hg 数据画在同一张图里
    for i in range(len(elelist)):
        plt.plot(wavnum_arr, ele_Deltakp_arr[i], label=elelist[i].name)
    # 加网格
    plt.grid(True)
    plt.show()

def GaussianShapeSquareDraw(b,elelist): #这个函数仿照2005年，注意画出的图是错的
    #b是国际单位制
    lenmax = 123.5
    lenmin = 121
    ele_Gbk_arr = []
    center_value = (wavlen2wavnum(lenmin) + wavlen2wavnum(lenmax))/2  # 中心波数
    print("中心波数为{}".format(center_value))
    #打印上限和下限
    print("波数上限{}".format(wavlen2wavnum(lenmin)))
    print("波数下限{}".format(wavlen2wavnum(lenmax)))
    length = 1000  # 数列长度,或者说数据点个数
    bandwidth = (wavlen2wavnum(lenmin) - wavlen2wavnum(lenmax))/2
    print("带宽为{}".format(bandwidth))

    # 生成中心值为某个波数的数列
    sequence = [center_value - bandwidth + i/length*2*bandwidth for i in range(length)]

    wavnum_arr = np.array(sequence)
    #换算成波长数组
    wavlen_arr = []
    for i,ele in enumerate(wavnum_arr):
        wavlen_arr.append(wavnum2wavlen(ele))    

    y1list = []
    y2list = []
    ylist = []

    for ele in elelist:
    #创建波数数组，范围为40000到60000，步长为100
        Gbk_arr = []
        if ele.name == "Hg" or ele.name == "Hg2":
            y1 = Delta_k(wavlen2roundfreq(257),ele)
            # print(wavlen2wavnum(257))
            # print(y1)
            # print(y1/ele.N * 10**4)
            print(Delta_k(wavenum2roundfreq(81150),ele))
        else:
            print("第一束基频光激光波长未定义，退出")
            exit()

        for wavnum in  wavnum_arr:
            y2 = Delta_k(wavenum2roundfreq(wavnum),ele) #这里使用Deltak是没有考虑原子密度N的影响
            y = y2 #国际单位的deltak,m-1，2005年论文很怪，按照他的图，omega1失谐不进行计算
            # 正确的y是y2-y1
            y1list.append(y1*b)
            y2list.append(y2*b)
            ylist.append(y*b)
            Gbk_arr.append(abs(GaussianShapeSquare(b,y))) 
        #打印y最大值
        print(max(y1list))
        print(max(y2list))
        ele_Gbk_arr.append(Gbk_arr)

    #Cd 和 Hg 数据画在同一张图里
    for i in range(len(elelist)):
        plt.plot(wavlen_arr, ele_Gbk_arr[i], label=elelist[i].name)
    # 加网格
    plt.grid(True)
    # x 和 y 轴的单位
    plt.xlabel('wavelength(nm)')
    plt.ylabel('Gaussian Shape Square(no unit)')
    plt.show()

def CompensateAngle(deltak, k1, k2, k3): #假设k1,k2相差不大
    theta = (deltak* (k1 + k2 + k3)/(k1 * k3))**0.5
    return theta

def CompensateDeltaK(k1, k2, k3, theta): #假设k1,k2相差不大
    cdk = k1 * k3/(k1 + k2 + k3) * (theta**2)
    return cdk

def rad2deg(rad):
    return rad * 180 / np.pi

def FWMout(P1, P2, P3, wavnum1, wavnum4, ExpSynsDict,Isotope,type, Pool:FWMpool,log = False,test = False,ShapeFactor = 0):
    #关闭打印
    if log== False:
        sys.stdout = open(os.devnull, 'w')

    print("四波混频池基本参数单位转换成cm")
    N_in_cm = Pool.N * (10**-6) #将原子密度转换为cm^-3
    L_in_cm = Pool.L * (10**2) #将腔长转换为cm
    A_in_cm = Pool.A * (10**4) #将腔面积转换为cm^2
    print("N_in_cm为{}".format(N_in_cm))
    print("L_in_cm为{}".format(L_in_cm))
    print("A_in_cm为{}".format(A_in_cm))

    print("开始计算四个圆频率")
    wavnum2 = abs(Pool.b - wavnum1)
    wavnum3 = abs(Pool.b - wavnum4)
    omega1, omega2, omega3, omega4 = wavenum2roundfreq(wavnum1), wavenum2roundfreq(wavnum2), wavenum2roundfreq(wavnum3), wavenum2roundfreq(wavnum4)

    print("开始计算非线性极化率")
    Kai1 = NLSusceptibility(wavnum1, Pool)
    Kai4 = NLSusceptibility(wavnum4, Pool)

    print("开始计算线型因子S")
    omegaD = DopplerBroadening(ExpSynsDict["omega0"],Pool) #到b态的doppler 展宽
    omegap = PressureBroadening(Pool)
    print("多普勒展宽为圆频率{}".format(omegaD))
    delta_wd = omegaD
    delta_wp = omegap
    print("气体压力展宽为圆频率{}".format(delta_wp))
    
    # omega_7s, gamma_gb = ExpSynsDict["omega_7s"], ExpSynsDict["gamma_gb"]
    #ShapeFactor = KaiShape(omega_7s,  delta_wd, ExpSynsDict["omega0"], 0,gamma_gb, delta_wp) #线型因子S,这里的omega0是中心圆频率（有2Pi），实验中是固定值，因为w1+w2必须和b能级共振
    # ShapeFactor = MultiIsotopeS( omega_7s, gamma_gb, delta_wp,Isotope,ExpSynsDict,Pool)
    # ShapeFactor2005 = MultiIsotopeS( omega_7s, gamma_gb, 2*constants.Pi *1.25*10**9,Isotope,ExpSynsDict,Pool)
    # if test == True:
    #     Kai_eff = Kai1 * Kai4 * ShapeFactor2005
    # else:
    Kai_eff = Kai1 * Kai4 * ShapeFactor
    print("Kai1为{}".format(Kai1))
    print("Kai4为{}".format(Kai4))
    print("ShapeFactor(单位cm)为{}".format(ShapeFactor))
    # print("ShapeFactor2005(单位cm)为{}".format(ShapeFactor2005))
    print("Kai_eff为{}".format(Kai_eff))

    print("开始计算phase matching的影响")
    y2 = Delta_k(wavenum2roundfreq(wavnum4),Pool) #这里使用Deltak是没有考虑原子密度N的影响
    y1 = Delta_k(wavenum2roundfreq(wavnum1),Pool) #这里使用Deltak是没有考虑原子密度N的影响
    ywrong = y2 #国际单位的deltak,m-1，2005年论文很怪，按照他的图，omega1失谐不进行计算
    ytrue = y2 - y1 #这个单位为m-1

    ywrong_in_cm = ywrong * (10**-2) #这个单位为cm-1
    ytrue_in_cm  = ytrue * (10**-2) #这个单位为cm-1

    print("正确的bdeltak为{}".format(Pool.confocal*ytrue))
    print("错误的bdeltak为{}".format(Pool.confocal*ywrong))
    curve = np.sin(ytrue_in_cm* L_in_cm /2) / (ytrue_in_cm* L_in_cm /2) 
    print("deltak的平行光curve为{}".format(curve))

    print("deltak(已经乘N)为{}".format(ytrue))
    print("开始计算补偿角")
    k1 = k_NL(omega=omega1, element=Pool)
    k3 = k_NL(omega=omega3, element=Pool)
    print("k1为{}".format(k1))
    print("k3为{}".format(k3))

    # if ytrue <= 0:
    #     compensatetheta = rad2deg(CompensateAngle(-ytrue,k1, k1, k3)) #代表补偿到完美的补偿角
    # else:
    #     compensatetheta = 180 #代表不可能补偿

    # print("补偿角为{}".format(compensatetheta))


    print("开始计算输出功率")

    if type == "Two Photon Resonance":
        k_coefficient = 1/4
    else:
        k_coefficient = 1

    
    #平行光功率计算
    p4parallel = 1.9 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * ((abs(Kai_eff* curve * N_in_cm * L_in_cm/A_in_cm* wavnum4))**2) # 注意这个地方应该是取模
    p4parallelperfect = 1.9 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * ((abs(Kai_eff* N_in_cm * L_in_cm/A_in_cm * wavnum4))**2) # 注意这个地方应该是取模

    print("平行光输出功率为{}".format(p4parallel))
    print("平行光完全没有失谐的功率为{}".format(p4parallelperfect)) #不推荐算平行光，因为A对量级的影响非常大

    #紧聚焦光功率计算

    
    #gfactor 和 ffactor互换
    if test == True:
        ffactorSquare = FShapeSquare(Pool.confocal,ywrong)
    else:
        ffactorSquare = FShapeSquare(Pool.confocal,ytrue)

    # import ipdb; ipdb.set_trace()
    print("ffactorSquare为{}".format(ffactorSquare))


    #p4tight = 7.8 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * wavnum1* wavnum2*wavnum3*wavnum4*((abs(Kai_eff* curve_tight*N_in_cm))**2) # 注意这个地方应该是取模
    p4tight = 7.8 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * wavnum1* wavnum2*wavnum3*wavnum4*((abs(Kai_eff* ffactorSquare**0.5*N_in_cm))**2) # 注意这个地方应该是取模
    p4tightperfect = 7.8 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * wavnum1* wavnum2*wavnum3*wavnum4*((abs(Kai_eff* 5.34**0.5*N_in_cm))**2) # 注意这个地方应该是取模

    print("紧聚焦光输出功率为{}".format(p4tight))
    print("紧聚焦光完全没有失谐的功率为{}".format(p4tightperfect)) #不推荐算平行光，因为A对量级的影响非常大

    bestN = Pool.N/(Pool.confocal*ytrue/(-4))
    print("best N is {}".format(bestN))

    if log== False:
        # reopen stdout file descriptor with write mode
        sys.stdout = sys.__stdout__
        
    return p4tight,p4tightperfect,bestN

def Delta_kp(omega,element:FWMpool): #这个函数输入的omega为波数
    # print("计算过程N为{}".format(element.N))
    # print("计算过程k为{}".format(k_NL(omega,element)+k_NL((element.b-omega),element)-2*k_NL(element.b/2,element)))
    # print("计算过程k1为{}".format(k_NL(omega,element)))
    # print("计算过程k2为{}".format(k_NL(element.b-omega,element)))
    # print("计算过程k3为{}".format(k_NL(element.b/2,element)))
    return  (k_NL(omega,element)+k_NL((wavenum2roundfreq(element.b)-omega),element)-2*k_NL(wavenum2roundfreq(element.b/2),element))/(element.N) 


def FWMout2005(P1, P2, P3, wavnum1, wavnum4, ExpSynsDict,type, Pool:FWMpool,log = False):
    #关闭打印
    if log== False:
        sys.stdout = open(os.devnull, 'w')

    print("四波混频池基本参数单位转换成cm")
    N_in_cm = Pool.N * (10**-6) #将原子密度转换为cm^-3
    L_in_cm = Pool.L * (10**2) #将腔长转换为cm
    A_in_cm = Pool.A * (10**4) #将腔面积转换为cm^2
    print("N_in_cm为{}".format(N_in_cm))
    print("L_in_cm为{}".format(L_in_cm))
    print("A_in_cm为{}".format(A_in_cm))

    print("开始计算四个圆频率")
    wavnum2 = abs(Pool.b - wavnum1)
    wavnum3 = abs(Pool.b - wavnum4)
    omega1, omega2, omega3, omega4 = wavenum2roundfreq(wavnum1), wavenum2roundfreq(wavnum2), wavenum2roundfreq(wavnum3), wavenum2roundfreq(wavnum4)

    print("开始计算非线性极化率")
    Kai1 = NLSusceptibility(wavnum1, Pool)
    Kai4 = NLSusceptibility(wavnum4, Pool)

    print("开始计算线型因子S")
    omegaD = DopplerBroadening(ExpSynsDict["omega0"],Pool) #到b态的doppler 展宽
    omegap = PressureBroadening(Pool)
    print("多普勒展宽为圆频率{}".format(omegaD))
    delta_wd = omegaD
    delta_wp = omegap
    print("气体压力展宽为圆频率{}".format(delta_wp))
    
    omega_7s, gamma_gb = ExpSynsDict["omega_7s"], ExpSynsDict["gamma_gb"]
    ShapeFactor = KaiShape(omega_7s,  delta_wd, ExpSynsDict["omega0"], 0,gamma_gb, delta_wp) #线型因子S,这里的omega0是中心圆频率（有2Pi），实验中是固定值，因为w1+w2必须和b能级共振
    Kai_eff = Kai1 * Kai4 * 8.026899902899876
    print("Kai1为{}".format(Kai1))
    print("Kai4为{}".format(Kai4))
    print("ShapeFactor(单位cm)为{}".format(ShapeFactor))
    print("Kai_eff为{}".format(Kai_eff))

    print("开始计算phase matching的影响")
    deltak1 = Delta_kp(omega1,Pool) * 10 **4 #单位换算到cm2
    deltak4 = Delta_kp(omega4,Pool) * 10 **4
    deltak = deltak4 - deltak1
    deltakN = deltak * N_in_cm


    print("开始计算补偿角")
    confocal_parameter = -2/(deltak * N_in_cm)
    curve = np.sin(deltakN* L_in_cm /2) / (deltakN * L_in_cm /2) 
    print("deltak的平行光curve为{}".format(curve))
    curve_tight = constants.Pi * confocal_parameter * deltak * N_in_cm * np.exp(confocal_parameter * deltak * N_in_cm/2)
    print("deltak的紧聚焦光curve为{}".format(curve_tight))

    print("deltak1为{}".format(deltak1))
    print("deltak4为{}".format(deltak4))
    print("deltak为{}".format(deltak))
    print("deltakN为{}".format(deltak*N_in_cm))

    print("开始计算输出功率")

    if type == "Two Photon Resonance":
        k_coefficient = 1/4
    else:
        k_coefficient = 1

    
    #平行光功率计算
    p4 = 1.9 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * ((abs(Kai_eff* curve * N_in_cm * L_in_cm/A_in_cm* wavnum4))**2) # 注意这个地方应该是取模
    p4perfect = 1.9 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * ((abs(Kai_eff* N_in_cm * L_in_cm/A_in_cm * wavnum4))**2) # 注意这个地方应该是取模

    print("平行光输出功率为{}".format(p4))
    print("平行光完全没有失谐的功率为{}".format(p4perfect)) #不推荐算平行光，因为A对量级的影响非常大

    #紧聚焦光功率计算

    y2 = Delta_k(wavenum2roundfreq(wavnum4),Pool) #这里使用Deltak是没有考虑原子密度N的影响
    y = y2 #国际单位的deltak,m-1，2005年论文很怪，按照他的图，omega1失谐不进行计算
    gfactor = abs(GaussianShapeSquare(0.0016,y))
    print("bdeltak为{}".format(0.0016*y))
    
    #gfactor 和 ffactor互换
    ffactor = gfactor/((0.0016*y)**2)
    print("ffactor为{}".format(ffactor))

    #p4tight = 7.8 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * wavnum1* wavnum2*wavnum3*wavnum4*((abs(Kai_eff* curve_tight*N_in_cm))**2) # 注意这个地方应该是取模
    p4tight = 7.8 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * wavnum1* wavnum2*wavnum3*wavnum4*((abs(Kai_eff* ffactor**0.5*N_in_cm))**2) # 注意这个地方应该是取模
    p4tightperfect = 7.8 * (10 ** -49)*k_coefficient *P1 * P2 * P3 * wavnum1* wavnum2*wavnum3*wavnum4*((abs(Kai_eff* 5.34**0.5*N_in_cm))**2) # 注意这个地方应该是取模

    print("紧聚焦光输出功率为{}".format(p4tight))
    print("紧聚焦光完全没有失谐的功率为{}".format(p4tightperfect)) #不推荐算平行光，因为A对量级的影响非常大

    return p4tight 

def AbsorbCrossSec(omega,omega0,gamma):
    return 3* constants.Pi * (constants.c**2)*(gamma**2)/(2*omega0**2)/((gamma/2)**2 + (omega - omega0)**2)

def AbsorbCrossSecDraw(Pool,isotopeinfo,plot):

    lifetime = 2e-9 #s
    gamma = 1/lifetime
          #创建波数数组，范围为40000到60000，步长为100
    S_arr = []

    center_value = 43692.48  # 中心值
    print("中心波长为{}".format(center_value))
    length = 1000  # 数列长度,或者说数据点个数
    bandwidth = 400

    # 生成中心值为某个波数的数列
    sequence = [center_value - i/length*2*bandwidth for i in range(length)]

    omega_arr = np.array(sequence)
    rfreq_arr = []
    #换算成圆频率数组
    for i,ele in enumerate(omega_arr):
        rfreq_arr.append(wavenum2roundfreq(ele))
    for rfreq in rfreq_arr:
            #从datta中取出同位素信息,每行读取遍历
        datas = 0
        for i in range(len(isotopeinfo)):
            omega_5p = isotopeinfo["omega"][i] +wavenum2roundfreq(-59219.73 +43692.38) #到a态的圆频率
            crosssec_iso = AbsorbCrossSec(rfreq,omega_5p,gamma)
            datas += (crosssec_iso) * isotopeinfo["abundance"][i]
        S_arr.append(datas)

    warnline = 1/Pool.N/1

    #将warnline画在图上

    if plot == True:
        #换算成波长
        wavlen_arr = []
        for i,ele in enumerate(omega_arr):
            wavlen_arr.append(wavnum2wavlen(ele))
        
        plt.plot(wavlen_arr, [warnline for i in range(len(omega_arr))],color = "red")
        plt.plot(wavlen_arr, S_arr)
        #显示出交点对应位置
        for i,ele in enumerate(S_arr):
            if ele < warnline:
                plt.scatter(wavlen_arr[i],ele)
        
        

        #坐标轴在10^-22左右
        plt.ylim(0,warnline)
        #纵坐标1/NL(SI unit)
        plt.ylabel("Absorption Cross Section(1/NL)")
        #横坐标(nm)
        plt.xlabel("wavelength(nm)")

        plt.show()
    elif plot == False:
        pass
    return omega_arr, S_arr

def AllShapeDraw(ExpSynsDict, Pool, Isotope,plot=False,test=True):
    Mwavnum_arr, Multi_arr = MultiShapeSquareGenerator(ExpSynsDict, Pool, Isotope,plot=plot,test=test)
    Swavnum_arr, Single_arr = SingleShapeSquareGenerator(ExpSynsDict, Pool, plot=plot,test=test)

    # Multi_arr,Single_arr画在一张图内
    plt.plot(Mwavnum_arr, Multi_arr, label="Multi")
    plt.plot(Swavnum_arr, Single_arr, label="Single")
    #根据同位素含量在对应波数位置画出线
    for i in range(len(Isotope)):
        plt.axvline(x=Isotope["centrewavnum"][i]+roundfreq2wavenum(ExpSynsDict["omega0"]),ymax=Isotope["abundance"][i], color='r', linestyle='-')
    #标题，mercury shape factor
    if Pool.name == "Cd":
        plt.title("Cadmium shape factor")
    plt.xlabel("wavnum")
    plt.ylabel("S^2 cm^2")
    plt.legend()
    plt.show()

def mk2wavnum(temperature): #国际单位输入
    energy = temperature * constants.k
    wavnum = roundfreq2wavenum(energy / constants.hbar)
    return wavnum

def wavnum2mk(wavnum): #国际单位输入
    energy = wavenum2roundfreq(wavnum) * constants.hbar
    temperature = energy / constants.k
    return temperature

#读取HgIsotope2020.csv文件,根据centrewavnum重新计算omega，保存
def HgIsotope2020(csvpath):
    HgIsotope2020 = pd.read_csv(csvpath)
    HgIsotope2020["omega"] = HgIsotope2020["centrewavnum"].apply(lambda x: wavenum2roundfreq(x+59219.734) )
    HgIsotope2020.to_csv(csvpath,index=False)

def KaiEffect4pic(wavnum1, wavnum4, Pool,Sfactor): #为了画图的函数
    kai_effect = NLSusceptibility(wavnum1, Pool)* NLSusceptibility(wavnum4, Pool) * Sfactor
    return abs(kai_effect)

def KaiEffectSI(wavnum1,wavnum4,Pool,Sfactor): #国际单位输入
    print(wavenum2roundfreq(1))
    kai1 = NLSusceptibility(wavnum1, Pool) * constants.ea0/wavenum2roundfreq(1)
    kai4 = NLSusceptibility(wavnum4, Pool) * constants.ea0/wavenum2roundfreq(1)
    return kai1*kai4*Sfactor #这个8是S因子

def ResonanceLineDraw(picaxis,csvpath): #绘制共振峰，红虚线 ,输入轴是波长
    picaxismax = np.max(picaxis)
    picaxismin = np.min(picaxis)
    #读取csv文件
    df = pd.read_csv(csvpath)
    #获取波数数组
    wavnum_arr = df['f_energy_wavnum']
    #转化为波长数组
    wavlen_arr = []
    for i,ele in enumerate(wavnum_arr):
        if picaxismin < wavnum2wavlen(ele) < picaxismax:
            wavlen_arr.append(wavnum2wavlen(ele))
            # 在该len位置绘制共振峰，红虚线
            plt.axvline(x=wavnum2wavlen(ele), color='r', linestyle='--')


def XAxisGenerator(lmin,lmax,datanumber): #绘制x轴，波长
    lenmin = lmin
    lenmax = lmax
    center_value = (wavlen2wavnum(lenmin) + wavlen2wavnum(lenmax))/2  # 中心波数
    print("中心波数为{}".format(center_value))
    #打印上限和下限
    print("波数上限{}".format(wavlen2wavnum(lenmin)))
    print("波数下限{}".format(wavlen2wavnum(lenmax)))
    length = datanumber # 数列长度,或者说数据点个数
    bandwidth = (wavlen2wavnum(lenmin) - wavlen2wavnum(lenmax))/2
    print("带宽为{}".format(bandwidth))

    # 生成中心值为某个波数的数列
    sequence = [center_value - bandwidth + i/length*2*bandwidth for i in range(length)]

    wavnum_arr = np.array(sequence)
    #换算成波长数组
    wavlen_arr = []
    for i,ele in enumerate(wavnum_arr):
        wavlen_arr.append(wavnum2wavlen(ele))
    return wavnum_arr,wavlen_arr

def Kaia3_2005_draw(PoolList,Sfactor):
    lenmax = 123.5
    lenmin = 121
    ele_kai_arr = []
    center_value = (wavlen2wavnum(lenmin) + wavlen2wavnum(lenmax))/2  # 中心波数
    print("中心波数为{}".format(center_value))
    #打印上限和下限
    print("波数上限{}".format(wavlen2wavnum(lenmin)))
    print("波数下限{}".format(wavlen2wavnum(lenmax)))
    length = 10000  # 数列长度,或者说数据点个数
    bandwidth = (wavlen2wavnum(lenmin) - wavlen2wavnum(lenmax))/2
    print("带宽为{}".format(bandwidth))

    wavnum1 = wavlen2wavnum(257)
    # 生成中心值为某个波数的数列
    sequence = [center_value - bandwidth + i/length*2*bandwidth for i in range(length)]

    wavnum_arr = np.array(sequence)
    #换算成波长数组
    wavlen_arr = []
    for i,ele in enumerate(wavnum_arr):
        wavlen_arr.append(wavnum2wavlen(ele))    

    for ele in PoolList:
    #创建波数数组，范围为40000到60000，步长为100
        Kaia3_arr = []
        for wavnum in  wavnum_arr:
            kaia3 = KaiEffect4pic(wavnum1,  wavnum, ele,Sfactor)
            Kaia3_arr.append(abs(kaia3)*10**6)
        ele_kai_arr.append(Kaia3_arr)

    plt.ylim(0, 5)
    # 设置y轴标签
    plt.ylabel('Kai_effective(10^-6)(ea0)^4')

    # Cd 和 Hg 数据画在同一张图里
    for i in range(len(PoolList)):
        plt.plot(wavlen_arr, ele_kai_arr[i], label=PoolList[i].name)
    #加图例，左上角
    plt.legend(loc='upper left')
    # 加网格
    plt.grid(True)
    # 加标题
    plt.title('Kai_effective of Hg(2005)')
    plt.show()

def Power_out_2005_draw(Hg63928ExpSynsDict, Hg202Isotope, ShapeFactor2005, HgPool):
    lenmax = 123.5
    lenmin = 121
    ele_Gbk_arr = []
    center_value = (wavlen2wavnum(lenmin) + wavlen2wavnum(lenmax))/2  # 中心波数
    print("中心波数为{}".format(center_value))
    #打印上限和下限
    print("波数上限{}".format(wavlen2wavnum(lenmin)))
    print("波数下限{}".format(wavlen2wavnum(lenmax)))
    length = 500  # 数列长度,或者说数据点个数
    bandwidth = (wavlen2wavnum(lenmin) - wavlen2wavnum(lenmax))/2
    print("带宽为{}".format(bandwidth))

    # 生成中心值为某个波数的数列
    sequence = [center_value - bandwidth + i/length*2*bandwidth for i in range(length)]

    wavnum_arr = np.array(sequence)
    #换算成波长数组
    wavlen_arr = []

    
    for i,wavnum in enumerate(wavnum_arr):
        wavlen_arr.append(wavnum2wavlen(wavnum))
    p_array = []
    f_array = []
    with tqdm(total=len(wavnum_arr)) as pbar:
        for i,wavnum4 in enumerate(wavnum_arr):
            p4tight,p4tightperfect, bestN = FWMout(P1=0.58, P2=0.5, P3=1.2, wavnum1 = wavlen2wavnum(257), wavnum4 = wavnum4, \
                                                            ExpSynsDict = Hg63928ExpSynsDict,Isotope = Hg202Isotope, type = "Three", \
                                                            Pool = HgPool,log = False,test = True, ShapeFactor = ShapeFactor2005)
            # p4tight = FWMout2005(P1=0.58, P2=0.5, P3=1.2, wavnum1 = wavlen2wavnum(257), wavnum4 = wavnum4, ExpSynsDict = Hg63928ExpSynsDict,type = "Three", Pool = HgPool,log = False)
            p_array.append(p4tight*10**6)
            y2 = Delta_k(wavenum2roundfreq(wavnum4),HgPool) #这里使用Deltak是没有考虑原子密度N的影响
            gfactor = abs(GaussianShapeSquare(0.0016,y2)) 
            #gfactor 和 ffactor互换
            ffactor = gfactor/((0.0016*y2)**2)
            f_array.append(ffactor)
            #f_array中等于0的点标红
            if ffactor == 0:
                ele_Gbk_arr.append(wavnum2wavlen(wavnum4))
            pbar.update(1)

   
    #绘图
    plt.plot(wavlen_arr,p_array)
    #标题 ，2005年的Hg 63928 Power Out
    plt.title("Hg 63928 Power Out(2005)")
    plt.scatter(ele_Gbk_arr,np.zeros(len(ele_Gbk_arr)),c = "r")
    plt.xlabel("wavelength(nm)")
    plt.ylabel("Power(uW)")
    # limit y scale
    plt.ylim(0, 1)

    plt.show()

