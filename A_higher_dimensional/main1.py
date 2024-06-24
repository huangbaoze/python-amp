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

class A_higher_dimensional:
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
        self.populationall = 5
        self.v = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        for i in range(self.populationall):
            #  矩阵
            self.v[i, :, :] = np.random.rand(2 * self.N + 1, 2 * self.N + 1) / 500 - 0.001  # -0.001~0.001
            # for j in range(1, self.N + 1, 2):
            # #间隔同环
            #     self.v[i, self.bandnum == j] = np.random.rand(1) / 500 - 0.001
            #  同环
            # matrixtmp = self.bandnum[self.bandnum >= 1]
            # self.v[i, self.bandnum >= 1] = (np.random.rand(156) / 500 - 0.001)[matrixtmp - 1]  # -0.001~0.001
        self.wmax = 0.9
        self.wmin = 0.4
        self.c1 = 2
        self.c2 = 3
        self.lam1 = 19
        self.lam2 = 10
        self.lam3 = 1
        self.iterationall = 500

    def run(self, mark, markk, markkk):
        bandminmax = np.zeros([2, self.N + 1])
        sizeboun = np.zeros(self.N + 1).astype(int)

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

            # 划定振幅优化实际范围
            bandminmax[0, i] = np.min(tmpA2)
            bandminmax[1, i] = np.max(tmpA2)

            # 计数
            H = bandA.shape[0]
            if i == 0:
                sizeboun[i] = H
                bandAall = np.zeros([int(sizeboun[i]), 3])
                bandPall = np.zeros([int(sizeboun[i]), 3])
                bandAall[:, 0] = tmpA1
                bandAall[:, 1] = tmpA2
                bandAall[:, 2] = tmpA3
                bandPall[:, 0] = tmpP1
                bandPall[:, 1] = tmpP2
                bandPall[:, 2] = tmpP3
            else:
                sizeboun[i] = H + sizeboun[i - 1]
                bandAall1 = np.zeros([int(sizeboun[i]), 3])
                bandPall1 = np.zeros([int(sizeboun[i]), 3])
                bandAall1[:, 0] = np.append(bandAall[:, 0], tmpA1)
                bandAall1[:, 1] = np.append(bandAall[:, 1], tmpA2)
                bandAall1[:, 2] = np.append(bandAall[:, 2], tmpA3)
                bandPall1[:, 0] = np.append(bandPall[:, 0], tmpP1)
                bandPall1[:, 1] = np.append(bandPall[:, 1], tmpP2)
                bandPall1[:, 2] = np.append(bandPall[:, 2], tmpP3)
                bandAall = bandAall1
                bandPall = bandPall1

        matrixband = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        matrixband1 = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        matrixband2 = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        matrixband3 = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap1 = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap2 = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap3 = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])

        # matrixband[0, self.bandnum >= 1] = 0.360911401039387
        # df = pd.DataFrame(matrixband[0].get())
        # df.to_excel('min_matrixband.xlsx', header=False, index=False)



        if mark == 0:
            for i in range(self.populationall):
                if markk == 0:
                    # 随机生成粒子（矩阵）
                    matrixtmp = self.bandnum[self.bandnum > 1]
                    arraysum1 = np.random.randint(sizeboun[matrixtmp - 2], sizeboun[matrixtmp - 1])
                    arraysum2 = np.random.randint(0, sizeboun[0])
                    matrixband[i, self.bandnum > 1] = bandAall[arraysum1, 1]
                    matrixband[i, self.bandnum == 1] = bandAall[arraysum2, 1]

                    matrixband1[i, self.bandnum > 1] = bandAall[arraysum1, 0]
                    matrixband1[i, self.bandnum == 1] = bandAall[arraysum2, 0]
                    matrixPhaseBandGap1[i, self.bandnum > 1] = bandPall[arraysum1, 0]
                    matrixPhaseBandGap1[i, self.bandnum == 1] = bandPall[arraysum2, 0]

                    matrixband2[i, :, :] = matrixband[i, :, :]
                    matrixPhaseBandGap2[i, self.bandnum > 1] = bandPall[arraysum1, 1]
                    matrixPhaseBandGap2[i, self.bandnum == 1] = bandPall[arraysum2, 1]

                    matrixband3[i, self.bandnum > 1] = bandAall[arraysum1, 2]
                    matrixband3[i, self.bandnum == 1] = bandAall[arraysum2, 2]
                    matrixPhaseBandGap3[i, self.bandnum > 1] = bandPall[arraysum1, 2]
                    matrixPhaseBandGap3[i, self.bandnum == 1] = bandPall[arraysum2, 2]

                    # 间隔同环
                    if markkk == 1:
                        for j in range(1, self.N + 1, 2):
                            if j == 1:
                                arraysum = np.random.randint(0, sizeboun[0])
                            else:
                                arraysum = np.random.randint(sizeboun[j - 2], sizeboun[j - 1])

                            matrixband[i, self.bandnum == j] = bandAall[arraysum, 1]

                            matrixband1[i, self.bandnum == j] = bandAall[arraysum, 0]
                            matrixPhaseBandGap1[i, self.bandnum == j] = bandPall[arraysum, 0]

                            matrixband2[i, :, :] = matrixband[i, :, :]
                            matrixPhaseBandGap2[i, self.bandnum == j] = bandPall[arraysum, 1]

                            matrixband3[i, self.bandnum == j] = bandAall[arraysum, 2]
                            matrixPhaseBandGap3[i, self.bandnum == j] = bandPall[arraysum, 2]

                if markk == 1:
                    # 随机生成粒子（同环）
                    arraysum = np.zeros(self.N + 1).astype(int)
                    arraysum[1: self.N + 1] = np.random.randint(sizeboun[np.arange(self.N)],
                                                                 sizeboun[np.arange(self.N) + 1])
                    arraysum[0] = np.random.randint(0, sizeboun[0])
                    matrixtmp = self.bandnum[self.bandnum >= 1]
                    matrixband[i, self.bandnum >= 1] = bandAall[arraysum[matrixtmp - 1], 1]

                    matrixband1[i, self.bandnum >= 1] = bandAall[arraysum[matrixtmp - 1], 0]
                    matrixPhaseBandGap1[i, self.bandnum >= 1] = bandPall[arraysum[matrixtmp - 1], 0]

                    matrixband2[i, self.bandnum >= 1] = bandAall[arraysum[matrixtmp - 1], 1]
                    matrixPhaseBandGap2[i, self.bandnum >= 1] = bandPall[arraysum[matrixtmp - 1], 1]

                    matrixband3[i, self.bandnum >= 1] = bandAall[arraysum[matrixtmp - 1], 2]
                    matrixPhaseBandGap3[i, self.bandnum >= 1] = bandPall[arraysum[matrixtmp - 1], 2]

        if mark == 1:
            min_band = pd.read_excel('E:\\huangbaoze\\matlab\\Amplitude optimization_upgrade_circular_symmetry_new_copy' +
                          '\\otherwavelengths_verification\\(54)_up_circle_28_19_10_discont_gd_gdd_37_4(end)\\min_band.xlsx', header=None, sheet_name=0).values
            min_band = np.array(min_band)
            snum = np.array([0, 5, 8, 2, 217])
            for i in range(self.populationall):
                for j in range(self.N + 1):
                    matrixband[i, self.bandnum == j + 1] = min_band[snum[i], self.N - j]
                    if j == 0:
                        matchingA2 = bandAall[0:sizeboun[0], 1]
                        matchingP2 = bandPall[0:sizeboun[0], 1]
                        matchingA1 = bandAall[0:sizeboun[0], 0]
                        matchingP1 = bandPall[0:sizeboun[0], 0]
                        matchingA3 = bandAall[0:sizeboun[0], 2]
                        matchingP3 = bandPall[0:sizeboun[0], 2]
                    else:
                        matchingA2 = bandAall[sizeboun[j - 1]:sizeboun[j], 1]
                        matchingP2 = bandPall[sizeboun[j - 1]:sizeboun[j], 1]
                        matchingA1 = bandAall[sizeboun[j - 1]:sizeboun[j], 0]
                        matchingP1 = bandPall[sizeboun[j - 1]:sizeboun[j], 0]
                        matchingA3 = bandAall[sizeboun[j - 1]:sizeboun[j], 2]
                        matchingP3 = bandPall[sizeboun[j - 1]:sizeboun[j], 2]

                    matchinggap = np.abs(matchingA2 - min_band[snum[i], self.N - j])
                    min_matchinggap = np.min(matchinggap, 0, keepdims=True)
                    matchingH = (matchinggap == min_matchinggap)
                    HTrue = np.where(matchingH == True)
                    if np.size(HTrue[0]) > 1:
                        matchingH[HTrue[0][0]] = False

                    matrixband2[i, self.bandnum == j + 1] = matchingA2[matchingH]
                    matrixPhaseBandGap2[i, self.bandnum == j + 1] = matchingP2[matchingH]
                    matrixband1[i, self.bandnum == j + 1] = matchingA1[matchingH]
                    matrixPhaseBandGap1[i, self.bandnum == j + 1] = matchingP1[matchingH]
                    matrixband3[i, self.bandnum == j + 1] = matchingA3[matchingH]
                    matrixPhaseBandGap3[i, self.bandnum == j + 1] = matchingP3[matchingH]


        # global record_matrixband, record_Ratio, record_Distance, record_Strength, record_FWHM, record_Dlimit, \
        #     min_matrixband, min_Ratio, min_Distance, min_Strength, min_FWHM, min_Dlimit

        min_matrixband = np.zeros([1, 2 * self.N + 1, 2 * self.N + 1])
        min_Ratio = np.zeros(1)
        min_Distance = np.zeros(1)
        min_Strength = np.zeros(1)
        min_FWHM = np.zeros(1)
        min_Dlimit = np.zeros(1)

        record_matrixband = np.zeros([self.populationall, 2 * self.N + 1, 2 * self.N + 1])
        record_Ratio = np.zeros(self.populationall)
        record_Distance = np.zeros(self.populationall)
        record_Strength = np.zeros(self.populationall)
        record_FWHM = np.zeros(self.populationall)
        record_Dlimit = np.zeros(self.populationall)

        Ratio = np.zeros(self.populationall)
        Distance = np.zeros(self.populationall)
        Strength = np.zeros(self.populationall)
        FWHM = np.zeros(self.populationall)
        Dlimit = np.zeros(self.populationall)

        # ite = []
        # index1 = []
        # index2 = []
        # index3 = []
        # index4 = []
        # index5 = []
        for iteration in range(self.iterationall):
            print(iteration)
            for PopulationNum in range(self.populationall):

                if iteration > 0:
                    w = self.wmax - (self.wmax - self.wmin) / (self.iterationall ** 2) * iteration ** 2
                    self.v[PopulationNum, self.bandnum >= 1] = w * self.v[PopulationNum, self.bandnum >= 1] + \
                                                               self.c1 * np.random.rand(1) * (record_matrixband[
                                                                                                  PopulationNum, self.bandnum >= 1] -
                                                                                              matrixband[
                                                                                                  PopulationNum, self.bandnum >= 1]) + \
                                                               self.c2 * np.random.rand(1) * (
                                                                       min_matrixband[0, self.bandnum >= 1] -
                                                                       matrixband[PopulationNum, self.bandnum >= 1])
                    # dff = pd.DataFrame(matrixband[PopulationNum].get())
                    # dff.to_excel(str(iteration) + 'matrixbandbefore' + str(PopulationNum) + '.xlsx',
                    #              sheet_name=str(iteration),
                    #              header=False, index=False)

                    matrixband[PopulationNum, self.bandnum >= 1] = matrixband[PopulationNum, self.bandnum >= 1] + \
                                                                   self.v[PopulationNum, self.bandnum >= 1]

                    tmpv1 = self.v[PopulationNum] > 0.01
                    self.v[PopulationNum, tmpv1] = 0.01
                    tmpv2 = self.v[PopulationNum] < -0.01
                    self.v[PopulationNum, tmpv2] = -0.01

                    matrixtmp = self.bandnum[self.bandnum >= 1]
                    tmpband1 = (matrixband[PopulationNum, self.bandnum >= 1] < bandminmax[0, matrixtmp - 1])

                    matrixbandtmp = matrixband[PopulationNum][self.bandnum >= 1]
                    matrixbandtmp[tmpband1] = bandminmax[0, matrixtmp[tmpband1] - 1]
                    matrixband[PopulationNum][self.bandnum >= 1] = matrixbandtmp

                    tmpband2 = (matrixband[PopulationNum, self.bandnum >= 1] > bandminmax[1, matrixtmp - 1])
                    matrixbandtmp[tmpband2] = bandminmax[1, matrixtmp[tmpband2] - 1]
                    matrixband[PopulationNum][self.bandnum >= 1] = matrixbandtmp
                    #
                    # df = pd.DataFrame(matrixband[PopulationNum].get())
                    # df.to_excel(str(iteration) + 'matrixband' + str(PopulationNum) + '.xlsx', sheet_name=str(iteration), header=False, index=False)

                    for t in range(self.N + 1):
                        if t == 0:
                            matchingA2 = bandAall[0:sizeboun[0], 1]
                            matchingP2 = bandPall[0:sizeboun[0], 1]
                            matchingA1 = bandAall[0:sizeboun[0], 0]
                            matchingP1 = bandPall[0:sizeboun[0], 0]
                            matchingA3 = bandAall[0:sizeboun[0], 2]
                            matchingP3 = bandPall[0:sizeboun[0], 2]
                        else:
                            matchingA2 = bandAall[sizeboun[t - 1]:sizeboun[t], 1]
                            matchingP2 = bandPall[sizeboun[t - 1]:sizeboun[t], 1]
                            matchingA1 = bandAall[sizeboun[t - 1]:sizeboun[t], 0]
                            matchingP1 = bandPall[sizeboun[t - 1]:sizeboun[t], 0]
                            matchingA3 = bandAall[sizeboun[t - 1]:sizeboun[t], 2]
                            matchingP3 = bandPall[sizeboun[t - 1]:sizeboun[t], 2]

                        matchinggap = np.abs(
                            matchingA2 - matrixband[PopulationNum, self.bandnum == t + 1].reshape(-1, 1))
                        min_matchinggap = np.min(matchinggap, 1, keepdims=True)
                        matchingH = (matchinggap == min_matchinggap)
                        # if t == 75:
                        #     df = pd.DataFrame(matchingH.get())
                        #     df.to_excel(str(iteration) + 'matchingH' + str(PopulationNum) + '.xlsx', sheet_name=str(iteration), header=False, index=False)
                        quantities = np.shape(matchingH)[0]
                        for tt in range(quantities):
                            HTrue = np.where(matchingH[tt] == True)
                            if np.size(HTrue[0]) > 1:
                                matchingH[tt, HTrue[0][0]] = False
                        matchingH1 = np.where(matchingH == True)[1]

                        matrixband[PopulationNum, self.bandnum == t + 1] = matchingA2[matchingH1]
                        matrixband2[PopulationNum, self.bandnum == t + 1] = matchingA2[matchingH1]
                        matrixPhaseBandGap2[PopulationNum, self.bandnum == t + 1] = matchingP2[matchingH1]
                        matrixband1[PopulationNum, self.bandnum == t + 1] = matchingA1[matchingH1]
                        matrixPhaseBandGap1[PopulationNum, self.bandnum == t + 1] = matchingP1[matchingH1]
                        matrixband3[PopulationNum, self.bandnum == t + 1] = matchingA3[matchingH1]
                        matrixPhaseBandGap3[PopulationNum, self.bandnum == t + 1] = matchingP3[matchingH1]


                ##后续这里可以把min_matchinggap画出来
                ##
                #################################

                ##角谱衍射
                ratio1, distance1, strength1 = Fun_diffraction(self.phi, matrixband3[PopulationNum],
                                                               matrixPhaseBandGap3[PopulationNum], self.lam3,
                                                               self.samplenum, self.Dx, self.T, self.N, self.bandnum,
                                                               self.lamc, self.lam)
                NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                DL1 = 0.5 / NA1
                fitness = 1 - (DL1 - ratio1)
                if fitness < 1:
                    ratio1, distance1, strength1 = Fun_diffraction(self.phi, matrixband2[PopulationNum],
                                                                   matrixPhaseBandGap2[PopulationNum], self.lam2,
                                                                   self.samplenum, self.Dx, self.T, self.N,
                                                                   self.bandnum, self.lamc, self.lam)
                    NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                    DL1 = 0.5 / NA1
                    fitness = 1 - (DL1 - ratio1)
                    fitness = fitness / 10
                    if fitness < 0.1:
                        ratio1, distance1, strength1 = Fun_diffraction(self.phi, matrixband1[PopulationNum],
                                                                       matrixPhaseBandGap1[PopulationNum], self.lam1,
                                                                       self.samplenum, self.Dx, self.T, self.N,
                                                                       self.bandnum, self.lamc, self.lam)
                        NA1 = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                        DL1 = 0.5 / NA1
                        fitness = 1 - (DL1 - ratio1)
                        fitness = fitness / 100

                Ratio[PopulationNum] = fitness
                Distance[PopulationNum] = distance1
                Strength[PopulationNum] = strength1
                FWHM[PopulationNum] = ratio1
                Dlimit[PopulationNum] = DL1

                if iteration > 0:
                    # print(f"上次迭代最佳{min_Ratio[0]}")
                    if Ratio[PopulationNum] < record_Ratio[PopulationNum]:
                        record_Ratio[PopulationNum] = Ratio[PopulationNum]
                        record_Distance[PopulationNum] = Distance[PopulationNum]
                        record_Strength[PopulationNum] = Strength[PopulationNum]
                        record_matrixband[PopulationNum, :, :] = matrixband[PopulationNum, :, :]
                        record_FWHM[PopulationNum] = FWHM[PopulationNum]
                        record_Dlimit[PopulationNum] = Dlimit[PopulationNum]

                    if record_Ratio[PopulationNum] < min_Ratio[0]:
                        # print(min_Ratio[0])
                        min_Ratio[0] = record_Ratio[PopulationNum]
                        min_Distance[0] = record_Distance[PopulationNum]
                        min_Strength[0] = record_Strength[PopulationNum]
                        min_matrixband[0] = record_matrixband[PopulationNum, :, :]
                        min_FWHM[0] = record_FWHM[PopulationNum]
                        min_Dlimit[0] = record_Dlimit[PopulationNum]
                        # print(min_Ratio[0])

            if iteration == 0:
                for j in range(self.populationall):
                    record_matrixband[j] = matrixband[j]
                    record_Ratio[j] = Ratio[j]
                    record_Distance[j] = Distance[j]
                    record_Strength[j] = Strength[j]
                    record_FWHM[j] = FWHM[j]
                    record_Dlimit[j] = Dlimit[j]

                location = np.where(record_Ratio == np.min(record_Ratio))  # 维度为一维[1,2,3]
                min_matrixband[0] = record_matrixband[location[0], :, :][0]
                min_Ratio[0] = record_Ratio[location][0]
                min_Distance[0] = record_Distance[location][0]
                min_Strength[0] = record_Strength[location][0]
                min_FWHM[0] = record_FWHM[location][0]
                min_Dlimit[0] = record_Dlimit[location][0]
                # print(min_Ratio[0])
                # print(min_Strength[0])

            data = np.array([[min_Ratio[0], min_Distance[0], min_Strength[0], min_FWHM[0], min_Dlimit[0]]])
            print(data.get())
            df1 = pd.DataFrame(data.get())
            df2 = pd.DataFrame(min_matrixband[0].get())
            global flag
            file1 = 'E:\\huangbaoze\\python\\A_higher_dimensional\\data\\' + str(flag) + '_min_Ratio.xlsx'
            file2 = 'E:\\huangbaoze\\python\\A_higher_dimensional\\data\\' + str(flag) + '_min_matrixband.xlsx'
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
                # with pd.ExcelWriter(file2, mode='a') as writer:
                #     df2.to_excel(writer, sheet_name=str(iteration), header=False, index=False)

            # ite.append(iteration)
            # index1.append(min_Ratio[0].get())
            # index2.append(min_Distance[0].get())
            # index3.append(min_Strength[0].get())
            # index4.append(min_FWHM[0].get())
            # index5.append(min_Dlimit[0].get())
            # plt.ion()
            # plt.figure(1)
            # plt.subplot(4, 1, 1)
            # plt.plot(ite, index1, marker='o')
            # plt.pause(0.1)
            # plt.subplot(4, 1, 2)
            # plt.plot(ite, index2, marker='.')
            # plt.pause(0.1)
            # plt.subplot(4, 1, 3)
            # plt.plot(ite, index3, marker='*')
            # plt.pause(0.1)
            # plt.subplot(4, 1, 4)
            # plt.plot(ite, index4, index5, marker='x')
            # plt.pause(0.1)
        return min_Ratio[0]


flag = 63
mark = 0  # 0随机（矩阵/同环），1同环固定初始
markk = 0  # 0随机（矩阵）1随机（同环）
markkk = 0  # (1)mark=0 (2)mark=0 (3)1间隔同环
if __name__ == '__main__':
    main = A_higher_dimensional()
    kk = main.run(mark, markk, markkk)
    while kk > 0.01:
        flag += 1
        main = A_higher_dimensional()
        kk = main.run(mark, markk, markkk)
