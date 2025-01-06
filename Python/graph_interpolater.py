import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pathlib
import matplotlib.pyplot as plt

INPUT_FILE = '20_percent_bsubpc.csv'
INPUT_FOLDER = 'Papers - Article OLEDs/Paper 1 - Data'

DESIRED_LOW_BOUND_X = 380
DESIRED_HIGH_BOUND_X = 780
DESIRED_STEP_X = 0.5
DESIRED_LOW_Y = 0
DESIRED_HIGH_Y = 0

def main():
	raw_data_folder_path = str(pathlib.Path().resolve()).replace("Python", INPUT_FOLDER)

	input_data = pd.read_csv(raw_data_folder_path+'/'+INPUT_FILE)
	print(input_data)

	column_names = input_data.columns

	if len(column_names) == 2:
		print("Two columns identified:", column_names)

		x_data = input_data.iloc[:, 0]
		y_data = input_data.iloc[:, 1]

		max_x_input = max(x_data)
		min_x_input = min(x_data)

		new_x_data = np.arange(DESIRED_LOW_BOUND_X, DESIRED_HIGH_BOUND_X, DESIRED_STEP_X)
		new_y_data = []

		for x_point in new_x_data:
			if x_point < min_x_input: 
				new_y_data.append(DESIRED_LOW_Y)
			elif x_point > max_x_input:
				new_y_data.append(DESIRED_HIGH_Y)
			else:
				above_index = next (x[0] for x in enumerate(x_data) if x[1] > x_point)
				
				x_2 = x_data[above_index]
				x_1 = x_data[above_index-1]
				y_2 = y_data[above_index]
				y_1 = y_data[above_index-1]

				m = (y_2 - y_1) / (x_2 - x_1)

				y_int = y_1 + m*(x_point - x_1)

				new_y_data.append(y_int)

		smoothed_spectra = savgol_filter(new_y_data, 51, 3)	

		#Graph of FD% vs FD_TOT
		fig, ax = plt.subplots()
		ax.scatter(x_data, y_data)
		#ax.scatter(new_x_data, new_y_data)
		ax.scatter(new_x_data, smoothed_spectra)
		ax.set_xlabel('Wavelength (nm)')
		ax.set_ylabel('Intensity')
		ax.set_title('Data Interpolation: Fluorescence')
		plt.savefig(raw_data_folder_path+'/'+INPUT_FILE.split('.')[0]+'_interpolation.png')

		output_df = pd.DataFrame ( {'x':new_x_data, 'y_int':new_y_data,'y_smoothed':smoothed_spectra} )
		print(output_df)
		output_df.to_csv(raw_data_folder_path+'/'+INPUT_FILE.split('.')[0]+'_output.csv', sep='\t', encoding='utf-8')



	else:
		print("Error: Program only can recognize two columns (x and y)")



if __name__ == '__main__':
    main()