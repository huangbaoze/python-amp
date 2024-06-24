try:
    import cupy as np

    print(f"cupy{np.__version__}")
except ModuleNotFoundError:
    print(ModuleNotFoundError)
    import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fun import Fun_diffraction
from openpyxl import load_workbook
import datetime

class A_higher_dimensional_DBS:
    def __init__(self):
        # unit: um
        self.lam = np.arange(68, 80 + 1 / 3, 1 / 3)
        self.lamc = self.lam[18]
        self.focallength = 315 * self.lamc
        self.T = 46.4
        self.N = 155
        self.matchingnum = 'matching_GD_GDD37'
        self.R = (self.N + 0.5) * self.T
        self.r = np.arange(0, (self.N + 1) * self.T, self.T)
        self.phi = 2 * np.pi / self.lamc * (self.focallength ** 2 - np.sqrt(self.r ** 2 + self.focallength ** 2))
        Num_L_W_A_P_beta = np.array(pd.read_excel('Num_L_W_A_P_betarad.xlsx', header=None, sheet_name=0))
        s = np.shape(Num_L_W_A_P_beta)[0]
        nn = np.floor(self.phi / (2 * np.pi))
        self.phi = self.phi - nn * 2 * np.pi
        aa = 0
        self.GeneNum = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            if self.phi[i] >= Num_L_W_A_P_beta[15, 4]:
                if np.abs(2 * np.pi - self.phi[i]) <= np.abs(self.phi[i] - Num_L_W_A_P_beta[15, 4]):
                    aa = 0
                else:
                    aa = 15
            else:
                for k in range(s - 1):
                    if Num_L_W_A_P_beta[k + 1, 4] > self.phi[i] >= Num_L_W_A_P_beta[k, 4]:
                        if np.abs(self.phi[i] - Num_L_W_A_P_beta[k, 4]) <= np.abs(
                                self.phi[i] - Num_L_W_A_P_beta[k + 1, 4]):
                            aa = k
                        else:
                            aa = k + 1
            self.phi[i] = Num_L_W_A_P_beta[aa, 4]
            self.GeneNum[i] = Num_L_W_A_P_beta[aa, 0]

        self.xx = np.ones((2 * self.N + 1, 1)) * np.arange(-self.N * self.T, (self.N + 1) * self.T, self.T)
        self.yy = (np.ones((2 * self.N + 1, 1)) * np.arange(self.N * self.T, -(self.N + 1) * self.T, -self.T)).T
        self.rr = np.sqrt(self.xx ** 2 + self.yy ** 2)
        # self.bandnum = np.floor(self.rr / self.T) + 1
        # self.bandnum[self.rr > self.R] = 0
        self.bandnum = np.array(pd.read_excel('E:\\huangbaoze\\python\\A_higher_dimensional\\accu_bandnum.xlsx', header=None, sheet_name=0).values)
        self.bandnum = self.bandnum.astype(int)
        self.samplenum = 4096
        self.Dx = 2 * self.R / self.samplenum
        # 超参数设置
        # self.populationall = 5
        # self.v = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        # for i in range(self.populationall):
        #     #  矩阵
        #     self.v[i, :, :] = np.random.rand(2 * self.N + 1, 2 * self.N + 1) / 500 - 0.001  # -0.001~0.001
        #     # for j in range(1, self.N + 1, 2):
        #     # #间隔同环
        #     #     self.v[i, self.bandnum == j] = np.random.rand(1) / 500 - 0.001
        #     #  同环
        #     # matrixtmp = self.bandnum[self.bandnum >= 1]
        #     # self.v[i, self.bandnum >= 1] = (np.random.rand(156) / 500 - 0.001)[matrixtmp - 1]  # -0.001~0.001
        # self.wmax = 0.9
        # self.wmin = 0.4
        # self.c1 = 2
        # self.c2 = 3
        self.lam1 = 10
        self.lam2 = 19
        self.lam3 = 28
        self.iterationall = 20000

    def run(self):
        bandminmax = np.zeros([3, 2, self.N + 1])
        phaseminmax = np.zeros([3, 2, self.N + 1])
        # sizeboun = np.zeros(self.N + 1).astype(int)

        global bandAall, bandPall
        for i in range(self.N + 1):
            file = 'E:\\huangbaoze\\matlab\\Start_HyperbolicLens\\' + self.matchingnum + '\\Lx_Ly_GD_GDD_rmse_A' + str(
                i + 1) + '.xlsx'
            bandA = pd.read_excel(file, header=None, sheet_name=1).values
            bandP = pd.read_excel(file, header=None, sheet_name=2).values
            bandA = np.array(bandA)
            bandP = np.array(bandP)

            tmpA1 = bandA[:, self.lam1 - 1]
            tmpA2 = bandA[:, self.lam2 - 1]
            tmpA3 = bandA[:, self.lam3 - 1]
            tmpP1 = bandP[:, self.lam1 - 1]
            tmpP2 = bandP[:, self.lam2 - 1]
            tmpP3 = bandP[:, self.lam3 - 1]

            sum1 = np.where(tmpA2 == np.min(tmpA2))[0][0]
            sum2 = np.where(tmpA2 == np.max(tmpA2))[0][0]
            # 划定振幅优化实际范围
            bandminmax[1, 0, i] = tmpA2[sum1]
            bandminmax[1, 1, i] = tmpA2[sum2]
            phaseminmax[1, 0, i] = tmpP2[sum1]
            phaseminmax[1, 1, i] = tmpP2[sum2]

            bandminmax[0, 0, i] = tmpA1[sum1]
            bandminmax[0, 1, i] = tmpA1[sum2]
            phaseminmax[0, 0, i] = tmpP1[sum1]
            phaseminmax[0, 1, i] = tmpP1[sum2]

            bandminmax[2, 0, i] = tmpA3[sum1]
            bandminmax[2, 1, i] = tmpA3[sum2]
            phaseminmax[2, 0, i] = tmpP3[sum1]
            phaseminmax[2, 1, i] = tmpP3[sum2]

        matrixband = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixband1 = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixband2 = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixband3 = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap1 = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap2 = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap3 = np.zeros([2 * self.N + 1, 2 * self.N + 1])

        # 随机生成粒子矩阵
        for j in range(self.N + 1):
            # 每环0/1随机
            matrixtmp = self.bandnum[self.bandnum == j + 1]
            arraysum = np.around(np.random.rand(np.size(matrixtmp))).astype(int)

            # # 同环0/1随机
            # arraysum = np.around(np.random.rand(1)).astype(int)

            matrixband[self.bandnum == j + 1] = bandminmax[1, arraysum, j]

            matrixband1[self.bandnum == j + 1] = bandminmax[0, arraysum, j]
            matrixPhaseBandGap1[self.bandnum == j + 1] = phaseminmax[0, arraysum, j]

            matrixband2[self.bandnum == j + 1] = bandminmax[1, arraysum, j]
            matrixPhaseBandGap2[self.bandnum == j + 1] = phaseminmax[1, arraysum, j]

            matrixband3[self.bandnum == j + 1] = bandminmax[2, arraysum, j]
            matrixPhaseBandGap3[self.bandnum == j + 1] = phaseminmax[2, arraysum, j]

        # 初始化
        # min_band = pd.read_excel('E:\\huangbaoze\\matlab\\Amplitude optimization_upgrade_circular_symmetry_new_copy' +
        #                          '\\otherwavelengths_verification\\(54)_up_circle_28_19_10_discont_gd_gdd_37_4(end)\\min_band.xlsx',
        #                          header=None, sheet_name=0).values
        # min_band = np.array(min_band)
        # for i in range(self.N + 1):
        #     print(i)
        #     matrixband[self.bandnum == i + 1] = min_band[217, self.N - i]
        #
        #     summ = np.where(bandminmax[1, :, i] == min_band[217, self.N - i])[0]
        #     if np.size(summ) == 0:
        #         summ = np.around(np.random.rand(1)).astype(int)
        #     print(summ)
        #
        #     matrixband1[self.bandnum == i + 1] = bandminmax[0, summ, i]
        #     matrixPhaseBandGap1[self.bandnum == i + 1] = phaseminmax[0, summ, i]
        #
        #     matrixband2[self.bandnum == i + 1] = bandminmax[1, summ, i]
        #     matrixPhaseBandGap2[self.bandnum == i + 1] = phaseminmax[1, summ, i]
        #
        #     matrixband3[self.bandnum == i + 1] = bandminmax[2, summ, i]
        #     matrixPhaseBandGap3[self.bandnum == i + 1] = phaseminmax[2, summ, i]


        Ratio = np.zeros(1)
        Distance = np.zeros(1)
        Strength = np.zeros(1)
        FWHM = np.zeros(1)
        Dlimit = np.zeros(1)
        min_matrixband = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        Efficiency = np.zeros(1)
        ItotalDisplay = np.zeros(1)
        ItotalDisplay_incident = np.zeros(1)

        ringband = 1
        star = 0
        global mmatrixbandtmp, mmatrixbandtmp1, mmatrixbandtmp2, mmatrixbandtmp3, mmatrixPhaseBandGaptmp1, \
            mmatrixPhaseBandGaptmp2, mmatrixPhaseBandGaptmp3
        for iteration in range(self.iterationall):
            print(iteration)
            if iteration > 0:
                # 每环0/1
                if star == np.size(matrixband[self.bandnum == ringband]):
                    star = 0
                    ringband += 1

                # 同环0/1
                # if ringband == self.N + 2:
                #     ringband = 1

                matrixbandtmp = matrixband[self.bandnum == ringband]
                mmatrixbandtmp = matrixband[self.bandnum == ringband]

                matrixbandtmp1 = matrixband1[self.bandnum == ringband]
                mmatrixbandtmp1 = matrixband1[self.bandnum == ringband]
                matrixPhaseBandGaptmp1 = matrixPhaseBandGap1[self.bandnum == ringband]
                mmatrixPhaseBandGaptmp1 = matrixPhaseBandGap1[self.bandnum == ringband]

                matrixbandtmp2 = matrixband2[self.bandnum == ringband]
                mmatrixbandtmp2 = matrixband2[self.bandnum == ringband]
                matrixPhaseBandGaptmp2 = matrixPhaseBandGap2[self.bandnum == ringband]
                mmatrixPhaseBandGaptmp2 = matrixPhaseBandGap2[self.bandnum == ringband]

                matrixbandtmp3 = matrixband3[self.bandnum == ringband]
                mmatrixbandtmp3 = matrixband3[self.bandnum == ringband]
                matrixPhaseBandGaptmp3 = matrixPhaseBandGap3[self.bandnum == ringband]
                mmatrixPhaseBandGaptmp3 = matrixPhaseBandGap3[self.bandnum == ringband]

                sum = 1 - np.where(bandminmax[1, :, ringband - 1] == matrixbandtmp[star])[0]
                if np.size(sum) == 0:
                    sum = np.around(np.random.rand(1)).astype(int)
                # 每环star 同环star:
                matrixbandtmp[star] = bandminmax[1, sum, ringband - 1]
                matrixband[self.bandnum == ringband] = matrixbandtmp

                matrixbandtmp1[star] = bandminmax[0, sum, ringband - 1]
                matrixband1[self.bandnum == ringband] = matrixbandtmp1
                matrixPhaseBandGaptmp1[star] = phaseminmax[0, sum, ringband - 1]
                matrixPhaseBandGap1[self.bandnum == ringband] = matrixPhaseBandGaptmp1

                matrixbandtmp2[star] = bandminmax[1, sum, ringband - 1]
                matrixband2[self.bandnum == ringband] = matrixbandtmp2
                matrixPhaseBandGaptmp2[star] = phaseminmax[1, sum, ringband - 1]
                matrixPhaseBandGap2[self.bandnum == ringband] = matrixPhaseBandGaptmp2

                matrixbandtmp3[star] = bandminmax[2, sum, ringband - 1]
                matrixband3[self.bandnum == ringband] = matrixbandtmp3
                matrixPhaseBandGaptmp3[star] = phaseminmax[2, sum, ringband - 1]
                matrixPhaseBandGap3[self.bandnum == ringband] = matrixPhaseBandGaptmp3

            ##角谱衍射
            ratio1, distance1, strength1, efficiency, ItotalDisplay_sum, ItotalDisplay_incident_sum = Fun_diffraction(self.phi, matrixband3,
                                                           matrixPhaseBandGap3, self.lam3,
                                                           self.samplenum, self.Dx, self.T, self.N, self.bandnum,
                                                           self.lamc, self.lam)
            NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
            DL1 = 0.5 / NA1
            fitness = 1 - (DL1 - ratio1)
            if fitness < 1:
                ratio1, distance1, strength1, efficiency, ItotalDisplay_sum, ItotalDisplay_incident_sum = Fun_diffraction(self.phi, matrixband2,
                                                               matrixPhaseBandGap2, self.lam2,
                                                               self.samplenum, self.Dx, self.T, self.N,
                                                               self.bandnum, self.lamc, self.lam)
                NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                DL1 = 0.5 / NA1
                fitness = 1 - (DL1 - ratio1)
                fitness = fitness / 10
                if fitness < 0.1:
                    ratio1, distance1, strength1, efficiency, ItotalDisplay_sum, ItotalDisplay_incident_sum = Fun_diffraction(self.phi, matrixband1,
                                                                   matrixPhaseBandGap1, self.lam1,
                                                                   self.samplenum, self.Dx, self.T, self.N,
                                                                   self.bandnum, self.lamc, self.lam)
                    NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                    DL1 = 0.5 / NA1
                    fitness = 1 - (DL1 - ratio1)
                    fitness = fitness / 100

            if iteration == 0:
                Ratio[0] = fitness
                Distance[0] = distance1
                Strength[0] = strength1
                FWHM[0] = ratio1
                Dlimit[0] = DL1
                Efficiency[0] = efficiency
                ItotalDisplay[0] = ItotalDisplay_sum
                ItotalDisplay_incident[0] = ItotalDisplay_incident_sum
                min_matrixband[:] = matrixband

            if iteration > 0:
                if fitness < Ratio[0]:
                    Ratio[0] = fitness
                    Distance[0] = distance1
                    Strength[0] = strength1
                    FWHM[0] = ratio1
                    Dlimit[0] = DL1
                    Efficiency[0] = efficiency
                    ItotalDisplay[0] = ItotalDisplay_sum
                    ItotalDisplay_incident[0] = ItotalDisplay_incident_sum
                    min_matrixband[:] = matrixband
                else:
                    matrixband[self.bandnum == ringband] = mmatrixbandtmp
                    matrixband1[self.bandnum == ringband] = mmatrixbandtmp1
                    matrixband2[self.bandnum == ringband] = mmatrixbandtmp2
                    matrixband3[self.bandnum == ringband] = mmatrixbandtmp3
                    matrixPhaseBandGap1[self.bandnum == ringband] = mmatrixPhaseBandGaptmp1
                    matrixPhaseBandGap2[self.bandnum == ringband] = mmatrixPhaseBandGaptmp2
                    matrixPhaseBandGap3[self.bandnum == ringband] = mmatrixPhaseBandGaptmp3
            # 每环
            star += 1

            # 同环
            # ringband += 1

            data = np.array([[Ratio[0], Distance[0], Strength[0], FWHM[0], Dlimit[0], Efficiency[0],  ItotalDisplay[0],  ItotalDisplay_incident[0]]])
            print(data.get())
            df1 = pd.DataFrame(data.get())
            df2 = pd.DataFrame(matrixband.get())
            global flag
            file1 = 'E:\\huangbaoze\\python\\A_higher_dimensional_DBS\\data\\' + str(flag) + '_min_Ratio.xlsx'
            file2 = 'E:\\huangbaoze\\python\\A_higher_dimensional_DBS\\data\\' + str(flag) + '_min_matrixband.xlsx'
            df2.to_excel(file2, header=False, index=False)
            if iteration == 0:
                df1.to_excel(file1, sheet_name='Sheet1', startrow=iteration, header=False, index=False)
                # df2.to_excel(file2, sheet_name=str(iteration), header=False, index=False)
            else:
                book = load_workbook(file1)
                with pd.ExcelWriter(file1) as writer:
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    df1.to_excel(writer, sheet_name='Sheet1', startrow=iteration, header=False, index=False)
        return Ratio[0]


flag = 14
if __name__ == '__main__':
    main = A_higher_dimensional_DBS()
    kk = main.run()
    while kk > 0.01:
        flag += 1
        main = A_higher_dimensional_DBS()
        kk = main.run()