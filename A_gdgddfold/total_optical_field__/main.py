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
import os

class PropagationPlot:
    def __init__(self):
        # unit: um
        self.lam = np.arange(8, 12 + 0.1, 0.1)
        self.lamc = self.lam[20]
        self.focallength = 315 * self.lamc
        C = 2.99792458 * 10 ** 14 / 1e12  # um/ps 光速
        self.w0 = 2 * np.pi * C / self.lamc
        self.w = 2 * np.pi * C / self.lam
        self.f = self.focallength / self.w0 ** 2 * self.w ** 2
        self.T = 7.25
        self.N_part = 99
        self.R = 200 * self.lamc  # 设定半径
        self.N = np.floor((self.R - self.T / 2) / self.T)
        self.N = int(self.N)
        print(self.N)
        self.matchingnum = 'fold_matching_GD_GDD1'
        # self.fold_num = np.floor((self.N + 1) / (self.N_part + 1))
        # self.fold_num = int(self.fold_num)
        self.r = np.arange(0, (self.N + 1) * self.T, self.T)
        self.phi = 2 * np.pi / self.lamc * (self.focallength ** 2 - np.sqrt(self.r ** 2 + self.focallength ** 2))
        Num_L_W_A_P_beta = np.array(
            pd.read_excel('D:\\zhaofen\\huangbaoze\\A_gdgddfold\\Num_L_W_A_P_betarad.xlsx', header=None, sheet_name=0))
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
        self.bandnum[self.rr > self.R] = 0
        self.bandnum = self.bandnum.astype(int)
        self.samplenum = 4096
        self.R = (self.N + 0.5) * self.T
        self.Dx = 2 * self.R / self.samplenum
        self.lamnum = 41

    def run(self, flag):
        sizeboun = np.zeros(self.N + 1).astype(int)
        global bandAall, bandPall
        for i in range(self.N + 1):
            file = 'D:\\zhaofen\\huangbaoze\\A_gdgddfold\\matching\\' + self.matchingnum + '\\Lx_Ly_GD_GDD_rmse_A' + str(
                i + 1) + '.xlsx'
            bandA = pd.read_excel(file, header=None, sheet_name=1).values
            bandP = pd.read_excel(file, header=None, sheet_name=2).values
            bandA = np.array(bandA)
            bandP = np.array(bandP)

            # 计数
            H = bandA.shape[0]
            if i == 0:
                sizeboun[i] = H
                bandAall = np.zeros([int(sizeboun[i]), self.lamnum])
                bandPall = np.zeros([int(sizeboun[i]), self.lamnum])
                for j in range(self.lamnum):
                    bandAall[:, j] = bandA[:, j]
                    bandPall[:, j] = bandP[:, j]
            else:
                sizeboun[i] = H + sizeboun[i - 1]
                bandAall1 = np.zeros([int(sizeboun[i]), self.lamnum])
                bandPall1 = np.zeros([int(sizeboun[i]), self.lamnum])
                for j in range(self.lamnum):
                    bandAall1[:, j] = np.append(bandAall[:, j], bandA[:, j])
                    bandPall1[:, j] = np.append(bandPall[:, j], bandP[:, j])
                bandAall = bandAall1
                bandPall = bandPall1

        matrixband = np.zeros([self.lamnum, 2 * self.N + 1, 2 * self.N + 1])
        matrixPhaseBandGap = np.zeros([self.lamnum, 2 * self.N + 1, 2 * self.N + 1])

        #0匹配排布
        if flag == 0:
            file = 'D:\\zhaofen\\huangbaoze\\total_optical_field\\otherwavelengths_verification\\A4_DBS_copy_(2)_37_19_1_(20000)\\2_min_matrixband.xlsx'
            matrixband0 = pd.read_excel(file, header=None, sheet_name=-1).values
            matrixband0 = np.array(matrixband0)
            for t in range(self.N + 1):
                if t == 0:
                    matchingA = np.zeros([int(sizeboun[t]), self.lamnum])
                    matchingP = np.zeros([int(sizeboun[t]), self.lamnum])
                    for k in range(self.lamnum):
                        matchingA[:, k] = bandAall[0:sizeboun[0], k]
                        matchingP[:, k] = bandPall[0:sizeboun[0], k]
                else:
                    matchingA = np.zeros([int(sizeboun[t] - sizeboun[t - 1]), self.lamnum])
                    matchingP = np.zeros([int(sizeboun[t] - sizeboun[t - 1]), self.lamnum])
                    for k in range(self.lamnum):
                        matchingA[:, k] = bandAall[sizeboun[t - 1]:sizeboun[t], k]
                        matchingP[:, k] = bandPall[sizeboun[t - 1]:sizeboun[t], k]
                matchinggap = np.abs(
                    matchingA[:, 18] - matrixband0[self.bandnum == t + 1].reshape(-1, 1))
                min_matchinggap = np.min(matchinggap, 1, keepdims=True)
                matchingH = (matchinggap == min_matchinggap)
                quantities = np.shape(matchingH)[0]
                for tt in range(quantities):
                    HTrue = np.where(matchingH[tt] == True)
                    if np.size(HTrue[0]) > 1:
                        matchingH[tt, HTrue[0][0]] = False
                matchingH = np.where(matchingH == True)[1]
                for s in range(self.lamnum):
                    matrixband[s, self.bandnum == t + 1] = matchingA[:, s][matchingH]
                    matrixPhaseBandGap[s, self.bandnum == t + 1] = matchingP[:, s][matchingH]
        #1随机排布
        if flag == 1:
            # 随机生成 每环
            # matrixtmp = self.bandnum[self.bandnum > 1]
            # arraysum1 = np.random.randint(sizeboun[matrixtmp - 2], sizeboun[matrixtmp - 1])
            # arraysum2 = np.random.randint(0, sizeboun[0])
            # for i in range(self.lamnum):
            #     print('波长'+str(i+1))
            #     matrixband[i, self.bandnum > 1] = bandAall[arraysum1, i]
            #     matrixband[i, self.bandnum == 1] = bandAall[arraysum2, i]
            #     matrixPhaseBandGap[i, self.bandnum > 1] = bandPall[arraysum1, i]
            #     matrixPhaseBandGap[i, self.bandnum == 1] = bandPall[arraysum2, i]

            #随机生成 同环
            for i in range(self.N + 1):
                if i == 0:
                    arraysum = np.random.randint(0, sizeboun[0])
                else:
                    arraysum = np.random.randint(sizeboun[i - 1], sizeboun[i])
                for j in range(self.lamnum):
                    matrixband[j, self.bandnum == i + 1] = bandAall[arraysum, j]
                    matrixPhaseBandGap[j, self.bandnum == i + 1] = bandPall[arraysum, j]

        global filenum
        path = 'D:/zhaofen/huangbaoze/A_gdgddfold/total_optical_field/otherwavelengths_verification/fold_random_' + str(
            filenum) + '/ZItotalDisplay'
        if not os.path.exists(path):
            os.makedirs(path)
            print("Folder created")
        else:
            print("Folder already exists")
        fold_phase = np.array(pd.read_excel('D:\\zhaofen\\huangbaoze\\A_gdgddfold\\optimise_3lam\\data\\1_min_fold_phase.xlsx', header=None, sheet_name=1).values)
        lac_num = np.array(pd.read_excel('D:\\zhaofen\\huangbaoze\\A_gdgddfold\\df_lac_num.xlsx', header=None, sheet_name=0).values)
        fold_phi = np.zeros(self.N + 1)
        for t in range(np.size(fold_phase[-1, ])):
            fold_phi[int(lac_num[0, t]) + 1:] = self.phi[int(lac_num[0, t]) + 1:] + fold_phase[-1, t]
        for i in range(self.lamnum):
            ratio1, distance1, strength1, SL, efficiency = Fun_diffraction(fold_phi, matrixband[i], matrixPhaseBandGap[i], i+1,
                                               self.samplenum, self.Dx, self.T, self.N, self.bandnum, self.lamc, self.lam, filenum)
            NA = np.sin(np.arctan(self.R / (distance1 * self.lamc)))
            DLWavelen = 0.5 / NA
            data = np.array([[DLWavelen[0], ratio1[0], self.f[i] / self.lamc, distance1[0], strength1, SL, efficiency]])
            file1 = 'D:\\zhaofen\\huangbaoze\\A_gdgddfold\\total_optical_field\\otherwavelengths_verification\\fold_random_' + str(filenum) + '\\DL_FWHM_f_FL_Ipeak.xlsx'
            df1 = pd.DataFrame(data.get())
            if i == 0:
                df1.to_excel(file1, sheet_name='Sheet1', startrow=i, header=False, index=False)
            else:
                book = load_workbook(file1)
                with pd.ExcelWriter(file1) as writer:
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    df1.to_excel(writer, sheet_name='Sheet1', startrow=i, header=False, index=False)
        file2 = 'D:\\zhaofen\\huangbaoze\\A_gdgddfold\\total_optical_field\\otherwavelengths_verification\\fold_random_' + str(filenum) + '\\min_matrixband.xlsx'
        df2 = pd.DataFrame(matrixband[20].get())
        df2.to_excel(file2, header=False, index=False)

filenum = 7
if __name__ == '__main__':
    while filenum < 9:
        main = PropagationPlot()
        main.run(1)
        filenum += 1




