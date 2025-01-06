import numpy as np
import pandas as pd
import pathlib
from numpy import dot
from numpy.linalg import inv 
from numpy.linalg import qr
import matplotlib.pyplot as plt
import math

COMBINED_SPECTRA = '20_percent_bsubpc_output.csv'
FD_SPECTRA = 'OLED-008_edited.csv'
OTHER_SPECTRA = 'AlQ3 only_output.csv'
INPUT_FOLDER = 'Papers - Article OLEDs/Paper 1 - Data'

X_HEADER_COMBO = 'x'
Y_HEADER_COMBO = 'y_smoothed'

X_HEADER_OTHER = 'x'
Y_HEADER_OTHER = 'y_smoothed'

X_HEADER_FD = 'Wavelength (nm)'
Y_HEADER_FD = 'Intensity (Counts)'


def main():
	folder_path = str(pathlib.Path().resolve()).replace("Python", INPUT_FOLDER)

	combined_spectra_df = pd.read_csv(folder_path+'/'+COMBINED_SPECTRA, delimiter='\t')
	fd_spectra_df = pd.read_csv(folder_path+'/'+FD_SPECTRA, delimiter='\t')
	other_spectra_df = pd.read_csv(folder_path+'/'+OTHER_SPECTRA, delimiter='\t')

	combo_x = combined_spectra_df[X_HEADER_COMBO].values
	combo_y = combined_spectra_df[Y_HEADER_COMBO].values

	fd_x = fd_spectra_df[X_HEADER_FD].values
	fd_y = fd_spectra_df[Y_HEADER_FD].values

	other_x = other_spectra_df[X_HEADER_OTHER].values
	other_y = other_spectra_df[Y_HEADER_OTHER].values

	fd_y_norm = np.array(fd_y)/np.sum(fd_y)
	other_y_norm = np.array(other_y)/np.sum(other_y)

	if np.all(combo_x == fd_x) and np.all(fd_x == other_x):
		print("Wavelengths are same!")

		A = np.array([fd_y_norm, other_y_norm]).T
		b = np.array(combo_y)
		
		print("A shape: ", A.shape)
		print("B shape: ", b.shape)

		x = dot(dot(inv(dot(A.T, A)),A.T),b)

		fd_percent = x[0]/(x[0]+x[1])
		print("Guess % FD = ", str(fd_percent*100)[0:5])


		simulated_spectra = x[0]*fd_y_norm + x[1]*other_y_norm

		corr_matrix = np.corrcoef(combo_y, simulated_spectra)
		corr = corr_matrix[0,1]
		R_sq = corr**2

		print("R-squared = ", R_sq)

		guess_fd_spectra = combo_y - x[1]*other_y_norm

		updated_fd_percent = np.sum(guess_fd_spectra) / (np.sum(guess_fd_spectra)+x[1])

		print("Updated Guess %FD", str(updated_fd_percent*100)[0:5])


		fig, ax = plt.subplots()
		ax.plot(combo_x, simulated_spectra.flatten())
		ax.plot(combo_x, combo_y.flatten())
		ax.plot(combo_x, guess_fd_spectra)
		ax.legend(["Simulated", "Actual", "Updated FD"])
		ax.set_xlabel('Wavelength (nm)')
		ax.set_ylabel('Intensity (Counts)')
		ax.set_title('Sample Deconvoluted Spectra')
		plt.savefig(folder_path+'/' + COMBINED_SPECTRA.split('.')[0]+'_deconvoluted.png')
		
	else:
		print("Wavelengths not the same")
if __name__ == '__main__':
    main()


