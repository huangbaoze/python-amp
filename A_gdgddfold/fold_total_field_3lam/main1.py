try:
    import cupy as np
    print(f"cupy{np.__version__}")
except ModuleNotFoundError:
    print(ModuleNotFoundError)
    import numpy as np
import pandas as pd
from openpyxl import load_workbook
from fun import Fun_diffraction

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
        self.fold_num = np.floor((self.N + 1) / (self.N_part + 1))
        self.fold_num = int(self.fold_num)
        self.r = np.arange(0, (self.N + 1) * self.T, self.T)
        self.phi = 2 * np.pi / self.lamc * (self.focallength ** 2 - np.sqrt(self.r ** 2 + self.focallength ** 2))
        Num_L_W_A_P_beta = np.array(pd.read_excel('E:\\huangbaoze\\python\\A_gdgddfold\\Num_L_W_A_P_betarad.xlsx', header=None, sheet_name=0))
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
        self.samplenum = 8192
        self.R = (self.N + 0.5) * self.T
        self.Dx = 2 * self.R / self.samplenum
        self.lamnum = 41

        ##GD/GDD分布
        self.coefficient_matrix = np.array(pd.read_excel('E:\\huangbaoze\\python\\A_gdgddfold\\coefficient_matrix.xlsx', header=None, sheet_name=0))
        self.populationall = 1
        self.iterationall = 500

    def run(self):

        fold_GD = np.zeros(self.N + 1)
        fold_GDD = np.zeros(self.N + 1)
        fold_GD[:] = self.coefficient_matrix[3, :]
        fold_GDD[:] = self.coefficient_matrix[2, :]

        fold_phase = np.zeros([self.populationall, self.fold_num])
        fold_phi = np.zeros([self.populationall, self.N + 1])
        for k in range(self.populationall):
            fold_phase[k] = 0 #np.random.rand(self.fold_num) * 2 * np.pi
            fold_phi[k] = self.phi

        for i in range(self.fold_num):
            if i < self.fold_num - 1:
                fold_GD[(i + 1) * (self.N_part + 1): (i + 2) * (self.N_part + 1)] = self.coefficient_matrix[3, (i + 1) *
                  (self.N_part + 1): (i + 2) * (self.N_part + 1)] - self.coefficient_matrix[3, (i + 1) * (self.N_part + 1)]
                fold_GDD[(i + 1) * (self.N_part + 1): (i + 2) * (self.N_part + 1)] = self.coefficient_matrix[2, (i + 1) *
                  (self.N_part + 1): (i + 2) * (self.N_part + 1)] - self.coefficient_matrix[2, (i + 1) * (self.N_part + 1)]
                for s in range(self.populationall):
                    fold_phi[s, (i + 1) * (self.N_part + 1): (i + 2) * (self.N_part + 1)] = self.phi[(i + 1) * (self.N_part + 1):
                      (i + 2) * (self.N_part + 1)] + fold_phase[s, i]

            if i == self.fold_num - 1:
                fold_GD[(i + 1) * (self.N_part + 1):] = self.coefficient_matrix[3, (i + 1) * (self.N_part + 1): ] - \
                                                           self.coefficient_matrix[3, (i + 1) * (self.N_part + 1)]
                fold_GDD[(i + 1) * (self.N_part + 1):] = self.coefficient_matrix[2, (i + 1) * (self.N_part + 1): ] - \
                                                           self.coefficient_matrix[2, (i + 1) * (self.N_part + 1)]
                for s in range(self.populationall):
                    fold_phi[s, (i + 1) * (self.N_part + 1):] = self.phi[(i + 1) * (self.N_part + 1):] + fold_phase[s, i]

        df1 = pd.DataFrame(fold_phi.get())
        df1.to_excel('fold_phi.xlsx', sheet_name='Sheet1', startrow=0, header=False, index=False)
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
                                                       - self.w0) + fold_GDD[n] * (self.w[lamnum[m] - 1] - self.w0) ** 2
                df2 = pd.DataFrame(matrixband[m].get())
                df3 = pd.DataFrame(matrixPhaseBandGap[m].get())
                if m == 0:
                    df2.to_excel('matrixband.xlsx', sheet_name=str(m), header=False, index=False)
                    df3.to_excel('matrixPhaseBandGap.xlsx', sheet_name=str(m), header=False, index=False)
                # else:
                #     with pd.ExcelWriter('matrixband.xlsx', mode='a') as writer:
                #         df2.to_excel(writer, sheet_name=str(m), header=False, index=False)
                    # with pd.ExcelWriter('matrixPhaseBandGap.xlsx', mode='a') as writer:
                    #     df3.to_excel(writer, sheet_name=str(m), header=False, index=False)

            for j in range(num):
                ##角谱衍射
                ratio1, distance1, strength1, SL, efficiency = Fun_diffraction(
                    fold_phi[l], matrixband[j], matrixPhaseBandGap[j], lamnum[j], self.samplenum, self.Dx, self.T, self.N, self.bandnum, self.lamc, self.lam)
                fitness = np.abs(distance1 - self.f[lamnum[j] - 1] / self.lamc) / (10 ** j)
                DLWavelen = 0.5 / np.sin(np.arctan(self.R / (distance1 * self.lamc)))
                print(distance1)
                print(DLWavelen)
                print(ratio1)
                print(strength1)
                print(j)
                data = np.array([[DLWavelen[0], ratio1[0], self.f[int(lamnum[j] - 1)] / self.lamc, distance1[0], strength1, SL, efficiency]])
                df4 = pd.DataFrame(data.get())
                if j == 0:
                    df4.to_excel('DL_FWHM_f_FL_Ipeak.xlsx', sheet_name='Sheet1', startrow=j, header=False, index=False)
                else:
                    book = load_workbook('DL_FWHM_f_FL_Ipeak.xlsx')
                    with pd.ExcelWriter('DL_FWHM_f_FL_Ipeak.xlsx') as writer:
                        writer.book = book
                        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                        df4.to_excel(writer, sheet_name='Sheet1', startrow=j, header=False, index=False)

                # if j == 0 and fitness > 10:
                #      break
                # if j == 0 and fitness < 10:
                #     continue
                # if j == 1 and fitness > 1:
                #     break
                # if j == 1 and fitness < 1:
                #     continue


if __name__ == '__main__':
    main = gd_gdd_fold()
    main.run()









