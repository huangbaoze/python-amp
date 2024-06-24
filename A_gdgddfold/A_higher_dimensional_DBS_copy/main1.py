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
        self.lam = np.arange(8, 12 + 0.1, 0.1)
        self.lamc = self.lam[20]
        self.focallength = 295 * self.lamc
        C = 2.99792458 * 10 ** 14 / 1e12  # um/ps 光速
        self.w0 = 2 * np.pi * C / self.lamc
        self.w = 2 * np.pi * C / self.lam
        self.f = self.focallength / self.w0 ** 2 * self.w ** 2
        self.T = 7.25
        self.N_part = 50
        self.R = 187 * self.lamc  # 设定半径
        self.N = np.floor((self.R - self.T / 2) / self.T)
        self.N = int(self.N)
        print(self.N)
        self.matchingnum = 'new_fold_matching_GD_GDD2'
        # self.fold_num = np.floor((self.N + 1) / (self.N_part + 1))
        # self.fold_num = int(self.fold_num)
        self.r = np.arange(0, (self.N + 1) * self.T, self.T)
        self.phi = 2 * np.pi / self.lamc * (self.focallength ** 2 - np.sqrt(self.r ** 2 + self.focallength ** 2))
        Num_L_W_A_P_beta = np.array(
            pd.read_excel('E:\\huangbaoze\\python\\A_gdgddfold\\Num_L_W_A_P_betarad.xlsx', header=None, sheet_name=0))
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
        self.bandnum = np.floor(self.rr / self.T) + 1
        self.R = (self.N + 0.5) * self.T
        self.bandnum[self.rr > self.R] = 0
        self.bandnum = self.bandnum.astype(int)
        self.samplenum = 8192
        self.Dx = 2 * self.R / self.samplenum
        self.lamcnum = 21
        self.lamnum = 41
        self.optimlam = np.array([41, 31, 21, 11, 1])
        # self.optimlam = np.array([1, 11, 21, 31, 41])
        self.optimlamnum = 5
        self.iterationall = 20000

    def run(self):
        bandminmax = np.zeros([5, 2, self.N + 1]) # 临时加上索引“3”代表第1波长，“4”代表第37波长
        phaseminmax = np.zeros([5, 2, self.N + 1])
        # sizeboun = np.zeros(self.N + 1).astype(int)

        # bandGDminmax = np.zeros([2, self.N + 1])
        # bandGDDminmax = np.zeros([2, self.N + 1])
        phaseminmax1 = np.zeros([5, 2, self.N + 1])
        global bandAall, bandPall
        for i in range(self.N + 1):
            file = 'E:\\huangbaoze\\python\\A_gdgddfold\\matching\\' + self.matchingnum + '\\Lx_Ly_GD_GDD_rmse_A' + str(
                i + 1) + '.xlsx'
            bandA = pd.read_excel(file, header=None, sheet_name=1).values
            bandP = pd.read_excel(file, header=None, sheet_name=2).values
            bandA = np.array(bandA)
            bandP = np.array(bandP)

            bandGDGDD = np.array(pd.read_excel(file, header=None, sheet_name=0).values[:, 2: 4])

            sum1 = np.where(bandA[:, self.lamcnum - 1] == np.min(bandA[:, self.lamcnum - 1]))[0][0]
            sum2 = np.where(bandA[:, self.lamcnum - 1] == np.max(bandA[:, self.lamcnum - 1]))[0][0]

            # bandGDminmax[0, i] = bandGDGDD[:, 0][sum1]
            # bandGDminmax[1, i] = bandGDGDD[:, 0][sum2]
            # bandGDDminmax[0, i] = bandGDGDD[:, 1][sum1]
            # bandGDDminmax[1, i] = bandGDGDD[:, 1][sum2]

            t = 0
            for lam in self.optimlam:
                bandminmax[t, 0, i] = bandA[:, lam - 1][sum1]
                bandminmax[t, 1, i] = bandA[:, lam - 1][sum2]
                phaseminmax[t, 0, i] = bandP[:, lam - 1][sum1]
                phaseminmax[t, 1, i] = bandP[:, lam - 1][sum2]
                phaseminmax1[t, 0, i] = bandGDGDD[:, 0][sum1] * (self.w[lam - 1] - self.w0) + bandGDGDD[:, 1][sum1] * (self.w[lam - 1] - self.w0) ** 2
                phaseminmax1[t, 1, i] = bandGDGDD[:, 0][sum2] * (self.w[lam - 1] - self.w0) + bandGDGDD[:, 1][sum2] * (self.w[lam - 1] - self.w0) ** 2
                t += 1

        matrixband = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        matrixband1 = np.zeros([self.optimlamnum, 2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap1 = np.zeros([self.optimlamnum, 2 * self.N + 1, 2 * self.N + 1])

        fold_phase = np.array(pd.read_excel('E:\\huangbaoze\\python\\A_gdgddfold\\optimise_3lam\\data\\19_min_fold_phase.xlsx', header=None, sheet_name=0).values)
        lac_num = np.array(pd.read_excel('E:\\huangbaoze\\python\\A_gdgddfold\\optimise_3lam\\new_fold\\df_lac_num.xlsx', header=None, sheet_name=0).values)
        fold_phi = np.zeros(self.N + 1)
        fold_phi[:] = self.phi
        for t in range(np.size(fold_phase[-1, ])):
            fold_phi[int(lac_num[0, t]) + 1:] = self.phi[int(lac_num[0, t]) + 1:] + fold_phase[-1, t]

        # 随机生成粒子矩阵
        for j in range(self.N + 1):
            # 每环0/1随机
            # matrixtmp = self.bandnum[self.bandnum == j + 1]
            # arraysum = np.around(np.random.rand(np.size(matrixtmp))).astype(int)

            # # 同环0/1随机
            arraysum = np.around(np.random.rand(1)).astype(int)

            matrixband[self.bandnum == j + 1] = bandminmax[2, arraysum, j]
            for k in range(self.optimlamnum):
                matrixband1[k, self.bandnum == j + 1] = bandminmax[k, arraysum, j]
                matrixPhaseBandGap1[k, self.bandnum == j + 1] = phaseminmax[k, arraysum, j]

        # ratio1, distance1, strength1, efficiency, ItotalDisplay_sum, ItotalDisplay_incident_sum = Fun_diffraction(
        #     fold_phi, matrixband1[0],
        #     matrixPhaseBandGap1[0], self.optimlam[0],
        #     self.samplenum, self.Dx, self.T, self.N, self.bandnum,
        #     self.lamc, self.lam, self.f / self.lamc)
        # NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
        # DL1 = 0.5 / NA1
        # fitness = (1 - (DL1 - ratio1)) / (10 ** 0)
        # if fitness > 1.03 or (np.abs(distance1 - self.f[int(self.optimlam[0]) - 1] / self.lamc) > 10):
        #     return 1


        # 初始化
        # min_band = pd.read_excel('E:\\huangbaoze\\python\\A_gdgddfold\\A_higher_dimensional_DBS_copy\\data\\7_min_matrixband - 243.xlsx',
        #                          header=None, sheet_name=0).values
        # min_band = np.array(min_band)
        # for i in range(self.N + 1):
        #     # matrixband[self.bandnum == i + 1] = min_band[self.N, self.N - i]
        #
        #     # summ = np.where(bandminmax[2, :, i] == min_band[self.N, self.N - i])[0][0]
        #     # print(min_band[self.bandnum == i+1])
        #     summ = np.where(bandminmax[2, :, i] == min_band[self.bandnum == i + 1][0])[0][0]
        #     # if np.size(summ) == 0:
        #     #     summ = np.around(np.random.rand(1)).astype(int)
        #
        #     matrixband[self.bandnum == i + 1] = bandminmax[2, summ, i]
        #
        #     for k in range(self.optimlamnum):
        #         matrixband1[k, self.bandnum == i + 1] = bandminmax[k, summ, i]
        #         matrixPhaseBandGap1[k, self.bandnum == i + 1] = phaseminmax[k, summ, i]


        Ratio = np.zeros(1)
        Distance = np.zeros(1)
        Strength = np.zeros(1)
        FWHM = np.zeros(1)
        Dlimit = np.zeros(1)
        min_matrixband = np.zeros([2 * self.N + 1, 2 * self.N + 1])
        Efficiency = np.zeros(1)
        ItotalDisplay = np.zeros(1)
        ItotalDisplay_incident = np.zeros(1)
        layers = np.zeros(1)

        ringband = 1
        star = 0
        for iteration in range(self.iterationall):
            print(iteration)
            if iteration > 0:
                # 每环0/1
                # if star == np.size(matrixband[self.bandnum == ringband]):
                #     star = 0
                #     ringband += 1

                # 同环0/1
                if ringband == self.N + 2:
                    ringband = 1

                matrixbandtmp = matrixband[self.bandnum == ringband]
                mmatrixbandtmp = matrixband[self.bandnum == ringband]

                matrixbandtmp1 = matrixband1[:, self.bandnum == ringband]
                mmatrixbandtmp1 = matrixband1[:, self.bandnum == ringband]
                matrixPhaseBandGaptmp1 = matrixPhaseBandGap1[:, self.bandnum == ringband]
                mmatrixPhaseBandGaptmp1 = matrixPhaseBandGap1[:, self.bandnum == ringband]

                sum = 1 - np.where(bandminmax[2, :, ringband - 1] == matrixbandtmp[star])[0][0]
                if np.size(sum) == 0:
                    sum = np.around(np.random.rand(1)).astype(int)
                # 每环star: star + 1 同环star:
                matrixbandtmp[star: ] = bandminmax[2, sum, ringband - 1]
                matrixband[self.bandnum == ringband] = matrixbandtmp

                matrixbandtmp1[:, star: ] = bandminmax[:, sum, ringband - 1].reshape(-1, 1)
                matrixband1[:, self.bandnum == ringband] = matrixbandtmp1
                matrixPhaseBandGaptmp1[:, star: ] = phaseminmax[:, sum, ringband - 1].reshape(-1, 1)
                matrixPhaseBandGap1[:, self.bandnum == ringband] = matrixPhaseBandGaptmp1

            for k in range(self.optimlamnum):
                ##角谱衍射
                ratio1, distance1, strength1, efficiency, ItotalDisplay_sum, ItotalDisplay_incident_sum = Fun_diffraction(fold_phi, matrixband1[k],
                                                               matrixPhaseBandGap1[k], self.optimlam[k],
                                                               self.samplenum, self.Dx, self.T, self.N, self.bandnum,
                                                               self.lamc, self.lam, self.f / self.lamc)
                NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                DL1 = 0.5 / NA1
                fitness = (1 - (DL1 - ratio1)) / (10 ** k)
                layers[0] = k
                if fitness > 1 / (10 ** k) or (np.abs(distance1 - self.f[int(self.optimlam[int(layers[0])]) - 1]/self.lamc) > 10):
                    break

            if iteration == 0:
                if (np.abs(distance1 - self.f[int(self.optimlam[int(layers[0])]) - 1]/self.lamc) > 10):
                    return 1
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
                if fitness < Ratio[0] and (np.abs(distance1 - self.f[int(self.optimlam[int(layers[0])]) - 1]/self.lamc) < 10):
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
                    matrixband1[:, self.bandnum == ringband] = mmatrixbandtmp1
                    # matrixband2[self.bandnum == ringband] = mmatrixbandtmp2
                    # matrixband3[self.bandnum == ringband] = mmatrixbandtmp3
                    matrixPhaseBandGap1[:, self.bandnum == ringband] = mmatrixPhaseBandGaptmp1
                    # matrixPhaseBandGap2[self.bandnum == ringband] = mmatrixPhaseBandGaptmp2
                    # matrixPhaseBandGap3[self.bandnum == ringband] = mmatrixPhaseBandGaptmp3
            # 每环
            # star += 1

            # 同环
            ringband += 1

            data = np.array([[Ratio[0], Distance[0], Strength[0], FWHM[0], Dlimit[0], Efficiency[0],  ItotalDisplay[0],  ItotalDisplay_incident[0]]])
            print(data.get())
            df1 = pd.DataFrame(data.get())
            df2 = pd.DataFrame(matrixband.get())
            global flag
            file1 = 'E:\\huangbaoze\\python\\A_gdgddfold\\A_higher_dimensional_DBS_copy\\data\\' + str(flag) + '_min_Ratio.xlsx'
            file2 = 'E:\\huangbaoze\\python\\A_gdgddfold\\A_higher_dimensional_DBS_copy\\data\\' + str(flag) + '_min_matrixband.xlsx'
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
    while kk == 1:
        main = A_higher_dimensional_DBS()
        kk = main.run()
    while kk > 0.0001:
        flag += 1
        main = A_higher_dimensional_DBS()
        kk = main.run()