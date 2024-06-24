try:
    import cupy as np
    print(f"cupy{np.__version__}")
except ModuleNotFoundError:
    print(ModuleNotFoundError)
    import numpy as np
import pandas as pd
from openpyxl import load_workbook
from fun import Fun_diffraction
import os

class gd_gdd_fold:
    def __init__(self):
        self.lam = np.arange(8, 12 + 0.1, 0.1)
        self.lamc = self.lam[20]
        self.focallength = 315 * self.lamc
        C = 2.99792458 * 10 ** 14 / 1e12  # um/ps 光速
        self.w0 = 2 * np.pi * C / self.lamc
        self.w = 2 * np.pi * C / self.lam
        self.f = self.focallength / self.w0 ** 2 * self.w ** 2
        self.T = 7.25
        self.N_part = 99
        self.R = 200 * self.lamc #设定半径
        self.N = np.floor((self.R - self.T / 2) / self.T)
        self.N = int(self.N)
        print(self.N)
        # self.fold_num = np.floor((self.N + 1) / (self.N_part + 1))
        # self.fold_num = int(self.fold_num)
        self.r = np.arange(0, (self.N + 1) * self.T, self.T)
        self.phi = 2 * np.pi / self.lamc * (self.focallength ** 2 - np.sqrt(self.r ** 2 + self.focallength ** 2))
        Num_L_W_A_P_beta = np.array(pd.read_excel('D:\\zhaofen\\huangbaoze\\A_gdgddfold\\Num_L_W_A_P_betarad.xlsx', header=None, sheet_name=0))
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
        self.samplenum = 2048
        self.Dx = 2 * self.R / self.samplenum
        self.lamnum = 41

        ##GD/GDD分布
        self.coefficient_matrix = np.array(pd.read_excel('D:\\zhaofen\\huangbaoze\\A_gdgddfold\\coefficient_matrix.xlsx', header=None, sheet_name=0))
        self.populationall = 1
        self.iterationall = 500

    def run(self):

        fold_GD = np.zeros(self.N + 1)
        fold_GDD = np.zeros(self.N + 1)
        fold_GD[:] = self.coefficient_matrix[3, :]
        fold_GDD[:] = self.coefficient_matrix[2, :]

        fold_num = 0
        flag = 1
        GD_fold_loc = self.N_part
        # GDD_fold_loc = self.N_part
        lac_num1 = np.zeros(0)

        global fold_phase1
        fold_phi = np.zeros([self.populationall, self.N + 1])
        for k in range(self.populationall):
            fold_phi[k] = self.phi

        optimise_fold_phase = np.array(pd.read_excel('D:\\zhaofen\\huangbaoze\\A_gdgddfold\\optimise_3lam\\data\\1_min_fold_phase.xlsx', header=None, sheet_name=1))
        while flag:
            fold_num += 1
            last_GD_fold_loc = GD_fold_loc
            # last_GDD_fold_loc = GDD_fold_loc

            GD_fold_loc = last_GD_fold_loc + np.where((self.coefficient_matrix[3, last_GD_fold_loc + 1:] - self.coefficient_matrix[3, last_GD_fold_loc + 1])
                                                <= self.coefficient_matrix[3, self.N_part])[0][-1] + 1
            # GDD_fold_loc = last_GDD_fold_loc + np.where(np.abs(self.coefficient_matrix[2, last_GDD_fold_loc + 1:] - self.coefficient_matrix[2, last_GDD_fold_loc + 1])
            #                                     <= self.coefficient_matrix[2, self.N_part])[0][-1] + 1
            # if GD_fold_loc <= GDD_fold_loc:
            #     GDD_fold_loc = GD_fold_loc
            # else:
            #     GD_fold_loc = GDD_fold_loc

            lac_num = np.zeros([1, fold_num])
            lac_num[0] = np.append(lac_num1, last_GD_fold_loc)
            lac_num1 = lac_num
            fold_GD[last_GD_fold_loc + 1: GD_fold_loc + 1] = self.coefficient_matrix[3, last_GD_fold_loc + 1: GD_fold_loc + 1] - \
                                                          self.coefficient_matrix[3, last_GD_fold_loc + 1]
            fold_GDD[last_GD_fold_loc + 1: GD_fold_loc + 1] = self.coefficient_matrix[2, last_GD_fold_loc + 1: GD_fold_loc + 1] - \
                                                         self.coefficient_matrix[2, last_GD_fold_loc + 1]

            # fold_phase_tmp = np.random.rand(self.populationall, 1) * 2 * np.pi
            fold_phase_tmp = np.zeros([self.populationall, 1])
            fold_phase_tmp[:, 0] = optimise_fold_phase[2, fold_num - 1]

            if fold_num == 1:
                fold_phase1 = np.zeros([self.populationall, 0])

            fold_phase = np.zeros([self.populationall, fold_num])
            for s in range(self.populationall):
                fold_phi[s, last_GD_fold_loc + 1: GD_fold_loc + 1] = self.phi[last_GD_fold_loc + 1: GD_fold_loc + 1] + fold_phase_tmp[s, 0]
                fold_phase[s] = np.append(fold_phase1[s], fold_phase_tmp[s, 0])
            fold_phase1 = fold_phase

            if GD_fold_loc == self.N:# or GDD_fold_loc == self.N:
                flag = 0
        global filenum
        path = 'D:/zhaofen/huangbaoze/A_gdgddfold/otherwavelengths_verification/fold_random_' + str(filenum) + '/ZItotalDisplay'
        if not os.path.exists(path):
            os.makedirs(path)
            print("Folder created")
        else:
            print("Folder already exists")
        fileall = 'D:\\zhaofen\\huangbaoze\\A_gdgddfold\\otherwavelengths_verification\\fold_random_' + str(filenum) + '\\'
        df_fold_GD = pd.DataFrame(fold_GD.get())
        df_fold_GDD = pd.DataFrame(fold_GDD.get())
        df_fold_phase = pd.DataFrame(fold_phase1.get())
        df_fold_GD.to_excel(fileall + 'df_fold_GD.xlsx', sheet_name='Sheet1', startrow=0, header=False, index=False)
        df_fold_GDD.to_excel(fileall + 'df_fold_GDD.xlsx', sheet_name='Sheet1', startrow=0, header=False, index=False)
        df_fold_phase.to_excel(fileall + 'df_fold_phase.xlsx', sheet_name='Sheet1', startrow=0, header=False, index=False)
        print(lac_num1.get())
        df1 = pd.DataFrame(fold_phi.get())
        df1.to_excel(fileall + 'fold_phi.xlsx', sheet_name='Sheet1', startrow=0, header=False, index=False)
        # lamnum = np.array([41, 21, 1])
        lamnum = np.arange(41) + 1
        num = 41
        for l in range(self.populationall):
            matrixband = np.zeros([num, 2 * self.N + 1, 2 * self.N + 1])
            matrixPhaseBandGap = np.zeros([num, 2 * self.N + 1, 2 * self.N + 1])
            for m in range(num):
                for n in range(self.N + 1):
                    matrixband[m, self.bandnum == n + 1] = 1
                    matrixPhaseBandGap[m, self.bandnum == n + 1] = fold_GD[n] * (self.w[lamnum[m] - 1]
                                                                                 - self.w0) + fold_GDD[n] * (
                                                                               self.w[lamnum[m] - 1] - self.w0) ** 2
                df2 = pd.DataFrame(matrixband[m].get())
                df3 = pd.DataFrame(matrixPhaseBandGap[m].get())
                if m == 0:
                    df2.to_excel(fileall + 'matrixband.xlsx', sheet_name=str(m), header=False, index=False)
                    df3.to_excel(fileall + 'matrixPhaseBandGap.xlsx', sheet_name=str(m), header=False, index=False)
                # else:
                #     with pd.ExcelWriter(fileall + 'matrixband.xlsx', mode='a') as writer:
                #         df2.to_excel(writer, sheet_name=str(m), header=False, index=False)
                # with pd.ExcelWriter(fileall + 'matrixPhaseBandGap.xlsx', mode='a') as writer:
                #     df3.to_excel(writer, sheet_name=str(m), header=False, index=False)

            for j in range(num):
                ##角谱衍射
                ratio1, distance1, strength1, SL, efficiency = Fun_diffraction(
                    fold_phi[l], matrixband[j], matrixPhaseBandGap[j], lamnum[j], self.samplenum, self.Dx, self.T,
                    self.N, self.bandnum, self.lamc, self.lam, filenum)
                # fitness = np.abs(distance1 - self.f[lamnum[j] - 1] / self.lamc) / (10 ** j)
                DLWavelen = 0.5 / np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                print(distance1)
                print(DLWavelen)
                print(ratio1)
                print(strength1)
                print(j)
                data = np.array([[DLWavelen[0], ratio1[0], self.f[lamnum[j] - 1] / self.lamc, distance1[0],
                                  strength1, SL, efficiency]])
                df4 = pd.DataFrame(data.get())
                if j == 0:
                    df4.to_excel(fileall + 'DL_FWHM_f_FL_Ipeak.xlsx', sheet_name='Sheet1', startrow=int(j), header=False, index=False)
                else:
                    book = load_workbook(fileall + 'DL_FWHM_f_FL_Ipeak.xlsx')
                    with pd.ExcelWriter(fileall + 'DL_FWHM_f_FL_Ipeak.xlsx') as writer:
                        writer.book = book
                        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                        df4.to_excel(writer, sheet_name='Sheet1', startrow=int(j), header=False, index=False)

                # if j == 0 and fitness > 10:
                #      break
                # if j == 0 and fitness < 10:
                #     continue
                # if j == 1 and fitness > 1:
                #     break
                # if j == 1 and fitness < 1:
                #     continue

filenum = 106
if __name__ == '__main__':
    while filenum < 107:
        main = gd_gdd_fold()
        main.run()
        filenum += 1










