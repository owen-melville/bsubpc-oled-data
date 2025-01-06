from os import path
import numpy as np
import pandas as pd
import pathlib
from numpy import dot
from numpy.linalg import inv 
from numpy.linalg import qr
import matplotlib.pyplot as plt
import math
import warnings
from matplotlib.patches import Rectangle
from openpyxl import load_workbook
from string import ascii_uppercase

OLED_OUTPUT_SHEET = 'OLED Output'
OLED_OUTPUT_COLUMNS = 'A:P'

OLED_DECONV_INPUT_SHEET = 'OLED Deconvolution'
OLED_DECONV_INPUT_COLUMNS = 'A:M'

MASTER_OLED_FILE = 'C:/Users/AM/Desktop/Offline Spreadsheets/Master OLED Log.xlsx'


WAVELENGTHS = np.arange(380,780,0.5)
#Reference spectral folder
REF_SPECTRAL_FOLDER = str(pathlib.Path().resolve()).replace("Python", "Reference Spectra") + "/" 
OLED_RAW_DATA_FOLDER = str(pathlib.Path().resolve()).replace("Python", "Raw OLED Data") + "/" 
OLED_OUTPUT_FOLDER = str(pathlib.Path().resolve()).replace("Python", "Output Data") + "/" 

OLED_PIXEL_NAMES = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5"]

def main():
	print("Running Spectral Deconvoluter")
	print("Created by Owen Melville, 2023")
	print("See Python Instructions File for Details")

	#Get Deposition Data
	spectral_input_df = get_offline_spreadsheet_as_df(MASTER_OLED_FILE, OLED_DECONV_INPUT_SHEET, OLED_DECONV_INPUT_COLUMNS)

	#Get OLED Summary Data
	oled_output_df = get_offline_spreadsheet_as_df(MASTER_OLED_FILE, OLED_OUTPUT_SHEET, OLED_OUTPUT_COLUMNS)

	warnings.filterwarnings("ignore")

	rows_to_edit = []
	for input_index, input_row in spectral_input_df.iterrows():
		if input_row['Status'] != 'Complete':
			#Get the device IDs for analysis of the HF device
			HF_device_id = input_row['HF Device ID']
			FD_device_id = input_row['FD Device ID']
			TD_device_id = input_row['TD Device ID']
			FD_name = input_row['FD Name']
			TD_name = input_row['TD Name']
			
			#Get the summary data for each device
			HF_output_row = oled_output_df[oled_output_df['OLED ID'].str.contains(HF_device_id)]
			FD_output_row = oled_output_df[oled_output_df['OLED ID'].str.contains(FD_device_id)]
			TD_output_row = oled_output_df[oled_output_df['OLED ID'].str.contains(TD_device_id)]

			print("Running Spectral Analysis: ", HF_device_id)

			#Create a directory if it doesn't exist
			local_deconv_save_path = OLED_OUTPUT_FOLDER +HF_device_id+ '/Deconvolution/'
			if path.exists(local_deconv_save_path):
			    print("Local output directory already exists (Deconvolution)...")
			else:
			    print("Creating Local Output Directory (Deconvolution): " +HF_device_id)
			    os.mkdir(local_deconv_save_path)

			#Process spectral data, make normalized "clean" spectra
			FD_file_name = input_row['Clean FD Spectral File']
			TD_file_name = input_row['Clean TD Spectral File']
			FD_spectral_pd = pd.read_csv(REF_SPECTRAL_FOLDER+FD_file_name, delimiter='\t')
			TD_spectral_pd = pd.read_csv(REF_SPECTRAL_FOLDER+TD_file_name, delimiter='\t')

			FD_spectral_array = np.array(FD_spectral_pd['Intensity (Counts)'].values)
			TD_spectral_array = np.array(TD_spectral_pd['Intensity (Counts)'].values)
			FD_normalized = FD_spectral_array/np.sum(FD_spectral_array)
			TD_normalized = TD_spectral_array/np.sum(TD_spectral_array)

			A = np.array([FD_normalized, TD_normalized]).T

			HF_color = HF_output_row['Median Color at 250 cd/m2'].values[0]
			full_HFQY_calculation = True
			try:
				TD_color = TD_output_row['Median Color at 250 cd/m2'].values[0]
				FD_color = FD_output_row['Median Color at 250 cd/m2'].values[0]
			except: #Let's say this isn't actually HFQY, just looking at amount of blue light
				TD_color = input_row['TD Device ID']
				FD_color = input_row['FD Device ID']
				full_HFQY_calculation = False


			device_raw_folder = OLED_RAW_DATA_FOLDER+HF_device_id+'/'
			td_raw_folder = OLED_RAW_DATA_FOLDER+TD_device_id+'/'
			fd_raw_folder = OLED_RAW_DATA_FOLDER+FD_device_id+'/'

			device_output_folder = OLED_OUTPUT_FOLDER+HF_device_id+'/Raw Data/'
			td_output_folder = OLED_OUTPUT_FOLDER+TD_device_id+'/Raw Data/'
			fd_output_folder = OLED_OUTPUT_FOLDER+FD_device_id+'/Raw Data/'

			hf_working_pixels = pd.read_csv(device_output_folder+HF_device_id+'_working_pixels.csv', delimiter='\t')['Pixel'].values
			if full_HFQY_calculation:
				td_working_pixels = pd.read_csv(td_output_folder +TD_device_id+'_working_pixels.csv', delimiter='\t')['Pixel'].values
				fd_working_pixels = pd.read_csv(fd_output_folder+FD_device_id+'_working_pixels.csv', delimiter='\t')['Pixel'].values


			hf_complete_data = pd.read_csv(device_output_folder+HF_device_id+'_all_optoelectronic.csv', delimiter='\t')
			if full_HFQY_calculation:
				fd_complete_data = pd.read_csv(fd_output_folder+FD_device_id+'_all_optoelectronic.csv', delimiter='\t')
				td_complete_data = pd.read_csv(td_output_folder+TD_device_id+'_all_optoelectronic.csv', delimiter='\t')

			#HF Data extraction
			num_data_points = hf_complete_data.shape[0]
			RAW_SPEC = np.empty([num_data_points, len(WAVELENGTHS)])

			count = 0
			for i in range (0, len(hf_working_pixels)):
				pixel = hf_working_pixels[i]
				raw_file_name = pixel + '_counts.txt'
				raw_summary_name = pixel + '.txt'
				raw_spectral_data = np.loadtxt(device_raw_folder+raw_file_name,delimiter="\t", dtype=float).T

				beta_fd = []
				beta_td = []
				fd_percent = []
				for j in range (2, len(raw_spectral_data)): #For each voltage, except the first 2 because there's 2 random extra 0 columns
					b = raw_spectral_data[j]
					b = np.array(b)[0:len(b)-1] #Off by 1 for some reason
					x = dot(dot(inv(dot(A.T, A)),A.T),b)
					beta_fd.append(x[0])
					beta_td.append(x[1])
					
					RAW_SPEC[count, :]=raw_spectral_data[j,0:len(WAVELENGTHS)]
					count=count+1
					
					if ((x[0]+x[1]) != 0):
						fd_percent.append(x[0]/(x[0]+x[1]))
					else:
						fd_percent.append(np.NaN)
			
				hf_complete_data.loc[hf_complete_data['Pixel']==pixel, 'FD_BETA']=beta_fd
				hf_complete_data.loc[hf_complete_data['Pixel']==pixel, 'TD_BETA']=beta_td
				hf_complete_data.loc[hf_complete_data['Pixel']==pixel, 'FD_PERCENT']=fd_percent

				#Lets compare our predicted spectra to our actual spectra
			
			simulated_spectra = np.outer(hf_complete_data['FD_BETA'],FD_normalized) + np.outer(hf_complete_data['TD_BETA'],TD_normalized)

			#Produce graph that compares raw to predicted data
			sample_index = next (x[0] for x in enumerate(hf_complete_data['L(cd/m2)'].values) if x[1] > 250)
			sample_simulated_spectra = simulated_spectra[ sample_index, :]
			sample_spectra = RAW_SPEC[sample_index , :]
			fig, ax = plt.subplots()
			ax.plot(WAVELENGTHS, sample_simulated_spectra.flatten())
			ax.plot(WAVELENGTHS, sample_spectra.flatten())
			ax.legend(["Simulated", "Actual"])
			ax.set_xlabel('Wavelength (nm)')
			ax.set_ylabel('Intensity (Counts)')
			ax.set_title('Sample Deconvoluted Spectra')
			plt.savefig(local_deconv_save_path+'Sample Deconvolution Spectra.png')

			#Processing the scatter data
			R_MATRIX = []
			for i in range (0, RAW_SPEC.shape[0]): #Devices
				corr_matrix = np.corrcoef(RAW_SPEC[i,:], simulated_spectra[i,:])
				corr = corr_matrix[0,1]
				R_sq = corr**2
				R_MATRIX.append(R_sq)

			hf_complete_data['R^2']=R_MATRIX

			lum_cutoff = 1
			hf_complete_data['Power Density (mW/cm2)']=hf_complete_data['Voltage(V)']*hf_complete_data['J(mA/cm2)']
			hf_cutoff_data = hf_complete_data[hf_complete_data['L(cd/m2)'] > lum_cutoff]

			fd_beta_scatter = hf_cutoff_data['FD_BETA'].values
			td_beta_scatter = hf_cutoff_data['TD_BETA'].values
			fd_comp_scatter = hf_cutoff_data['FD_PERCENT'].values
			power_density_hf_scatter =  hf_cutoff_data['Power Density (mW/cm2)'].values
			luminance_hf_scatter = hf_cutoff_data['L(cd/m2)'].values
			r2_scatter = hf_cutoff_data['R^2'].values

			LUMINANCE_THRESHOLD = 250 #Let's make this adjustable later
			LUMINANCE_PLUSMINUS = 50 #For calculating values within a region instead of at an exact value

			lum_bounded = hf_cutoff_data[hf_cutoff_data['L(cd/m2)'] > LUMINANCE_THRESHOLD - LUMINANCE_PLUSMINUS]
			lum_bounded = lum_bounded[lum_bounded['L(cd/m2)'] < LUMINANCE_THRESHOLD + LUMINANCE_PLUSMINUS]
			threshold_fdpercent = np.median(lum_bounded['FD_PERCENT'].values)
			print("Average %FD @ L = "+str(LUMINANCE_THRESHOLD)+"+/-" + str(LUMINANCE_PLUSMINUS)+" cd/m2: " + str(threshold_fdpercent*100)[0:5] + " %")
			threshold_r2 = np.median(lum_bounded['R^2'].values)
			print("Average R^2 @ L = "+str(LUMINANCE_THRESHOLD)+"+/-" + str(LUMINANCE_PLUSMINUS)+" cd/m2: " + str(threshold_r2)[0:5])

			POWER_THRESHOLD = 500 #Let's make this adjustable later
			POWER_PLUSMINUS = 50 #For calculating values within a region instead of at an exact value

			power_bounded = hf_cutoff_data[hf_cutoff_data['Power Density (mW/cm2)'] > POWER_THRESHOLD - POWER_PLUSMINUS]
			power_bounded = power_bounded[power_bounded['Power Density (mW/cm2)'] < POWER_THRESHOLD + POWER_PLUSMINUS]
			threshold_fd_total = np.median(power_bounded['FD_BETA'].values)
			print("Average FD Intensity @ P = "+str(POWER_THRESHOLD)+"+/-" + str(POWER_PLUSMINUS)+" mW/cm2: " + format_e(threshold_fd_total) + " counts")

			#Let's take lum_bounded and power_bounded and make visual reports and do analysis by thickness
			summary_df = create_visual_report(lum_bounded, power_bounded, HF_device_id, local_deconv_save_path, FD_color)
			check_correlations(summary_df,HF_device_id, OLED_OUTPUT_FOLDER+HF_device_id+'/Correlations/')

			#Graph of FD% vs FD_TOT
			fig, ax = plt.subplots()
			ax.scatter(summary_df['FD_ct'], summary_df['FD_%'])
			ax.set_xlabel('Average FD Intensity (Counts) at 500 mW/cm2'r'$\pm$'+'100 mW/cm2)')
			ax.set_ylabel('Percentage FD Emission (%) at 250 cd/m2')
			ax.set_title('FD Intensity vs Percentage Trade-off: ' + HF_device_id)
			plt.savefig(local_deconv_save_path+'FD_tradeoff.png')
			plt.clf()
			plt.close()

			fig, ax = plt.subplots()
			ax.scatter(luminance_hf_scatter, r2_scatter, color='blue')
			ax.set_xlabel('Luminance (cd/m2)')
			ax.set_ylabel('R Squared')
			ax.set_title('Deconvolution Error: ' + HF_device_id)
			plt.savefig(local_deconv_save_path+'Deconvolution Error.png')

			#Here we create the composition variation graphs... I've commented out the averaged versions
			fig, ax = plt.subplots()
			ax2=ax.twinx()
			ax.scatter(luminance_hf_scatter, fd_beta_scatter,color=FD_color)
			ax.scatter(luminance_hf_scatter, td_beta_scatter,color=TD_color)
			ax2.scatter(luminance_hf_scatter, fd_comp_scatter,color='blue')
			ax2.set_ylim([0,1])
			ax.set_xlabel('Luminance (cd/m2)')
			ax.set_ylabel('Intensity (Counts)')
			ax2.set_ylabel('% Fluorescent Dopant')
			ax.set_title('FD/TD Composition: ' + HF_device_id)
			#box = ax.get_position()
			#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			ax.set_xlim([0,None])
			ax.set_ylim([0,None])
			ax.legend(["FD: "+FD_name, "TD: "+TD_name],loc='center left', bbox_to_anchor=(0, 0.60))
			ax2.legend(["%FD"],loc='center left', bbox_to_anchor=(0, 0.425))
			plt.savefig(local_deconv_save_path+HF_device_id+'_deconvolution.png')
			plt.clf()
			plt.close()


			if full_HFQY_calculation:
				num_data_points = fd_complete_data.shape[0]
				FD_RAW_SPEC = np.empty([num_data_points, len(WAVELENGTHS)])
				count=0
				for i in range (0, len(fd_working_pixels)):
					pixel = fd_working_pixels[i]
					raw_file_name = pixel + '_counts.txt'
					raw_summary_name = pixel + '.txt'
					raw_spectral_data = np.loadtxt(fd_raw_folder+raw_file_name,delimiter="\t", dtype=float).T[2:, :] #Get rid of 0 columns
					for spectra in raw_spectral_data:
						FD_RAW_SPEC[count, :] = spectra[0:len(WAVELENGTHS)]
						count+=1

					raw_total_intensity = np.sum(raw_spectral_data, axis=1)
					fd_complete_data.loc[fd_complete_data['Pixel']==pixel, 'FD_TOT']=raw_total_intensity

				count=0
				num_data_points = td_complete_data.shape[0]
				TD_RAW_SPEC = np.empty([num_data_points, len(WAVELENGTHS)])	
				for i in range (0, len(td_working_pixels)):
					pixel = td_working_pixels[i]
					raw_file_name = pixel + '_counts.txt'
					raw_summary_name = pixel + '.txt'
					raw_spectral_data = np.loadtxt(td_raw_folder+raw_file_name,delimiter="\t", dtype=float).T[2:, :] #Get rid of 0 columns
					for spectra in raw_spectral_data:
						TD_RAW_SPEC[count, :] = spectra[0:len(WAVELENGTHS)]
						count+=1
					raw_total_intensity = np.sum(raw_spectral_data, axis=1)
					td_complete_data.loc[td_complete_data['Pixel']==pixel, 'TD_TOT']=raw_total_intensity

				fd_complete_data['Power Density (mW/cm2)']=fd_complete_data['Voltage(V)']*fd_complete_data['J(mA/cm2)']
				fd_cutoff_data = fd_complete_data[fd_complete_data['L(cd/m2)'] > lum_cutoff]

				td_complete_data['Power Density (mW/cm2)']=td_complete_data['Voltage(V)']*td_complete_data['J(mA/cm2)']
				td_cutoff_data = td_complete_data[td_complete_data['L(cd/m2)'] > lum_cutoff]

		
				fd_scatter = fd_cutoff_data['FD_TOT'].values
				power_density_fd_scatter =  fd_cutoff_data['Power Density (mW/cm2)'].values
				td_scatter = td_cutoff_data['TD_TOT'].values
				power_density_td_scatter =  td_cutoff_data['Power Density (mW/cm2)'].values

				#Here we plot the variation by power density
				fig, ax = plt.subplots()
				ax.scatter(power_density_hf_scatter, fd_beta_scatter,color=FD_color)
				ax.scatter(power_density_hf_scatter, td_beta_scatter,color=TD_color)
				ax.scatter(power_density_fd_scatter, fd_scatter,color=FD_color,facecolors='none')
				ax.scatter(power_density_td_scatter, td_scatter,color=TD_color,facecolors='none')
				ax.legend(["FD - HF OLED", "TD - HF OLED", "FD Only OLED", "TD Only OLED"])
				ax.set_xlim([0,None])
				ax.set_ylim([0,None])
				ax.set_xlabel('Power Density (mW/cm2)')
				ax.set_ylabel('Intensity (Counts)')
				ax.set_title('Comparison of Hyperfluorescent Device to TD/FD-Only Controls')
				plt.savefig(local_deconv_save_path+HF_device_id+'_TDFD_Comparison.png')		

				#Lets do the Boost and HFQY
				max_hf_power = max (power_density_hf_scatter)
				max_td_power = max (power_density_td_scatter)
				max_fd_power = max (power_density_fd_scatter)
				
				POWER_THRESHOLD = 500 #Let's make this adjustable later
				POWER_PLUSMINUS = 100 #For calculating values within a region instead of at an exact value

				#Lets get indices and make graphs to show hyperfluorescence
				hf_power_rows = hf_cutoff_data[(hf_cutoff_data['Power Density (mW/cm2)'] >= POWER_THRESHOLD-POWER_PLUSMINUS)&(hf_cutoff_data['Power Density (mW/cm2)'] <= POWER_THRESHOLD+POWER_PLUSMINUS)]
				fd_power_rows = fd_cutoff_data[(fd_cutoff_data['Power Density (mW/cm2)'] >= POWER_THRESHOLD-POWER_PLUSMINUS)&(fd_cutoff_data['Power Density (mW/cm2)'] <= POWER_THRESHOLD+POWER_PLUSMINUS)]
				td_power_rows = td_cutoff_data[(td_cutoff_data['Power Density (mW/cm2)'] >= POWER_THRESHOLD-POWER_PLUSMINUS)&(td_cutoff_data['Power Density (mW/cm2)'] <= POWER_THRESHOLD+POWER_PLUSMINUS)]

				hf_power_indices = hf_power_rows.index
				fd_power_indices = fd_power_rows.index
				td_power_indices = td_power_rows.index

				hf_avg_spectra = np.average(RAW_SPEC[hf_power_indices, :], axis=0)
				fd_avg_spectra = np.average(FD_RAW_SPEC[fd_power_indices, :], axis=0)
				td_avg_spectra = np.average(TD_RAW_SPEC[td_power_indices, :], axis=0)

				hf_avg_spectra_std = np.std(RAW_SPEC[hf_power_indices, :], axis=0)
				fd_avg_spectra_std = np.std(FD_RAW_SPEC[fd_power_indices, :], axis=0)
				td_avg_spectra_std = np.std(TD_RAW_SPEC[td_power_indices, :], axis=0)

				fig, ax = plt.subplots()

				ax.plot(WAVELENGTHS, hf_avg_spectra, color='red')
				ax.fill_between(WAVELENGTHS, hf_avg_spectra+hf_avg_spectra_std,hf_avg_spectra-hf_avg_spectra_std,alpha=0.3, color='red')
				ax.plot(WAVELENGTHS, fd_avg_spectra, color=FD_color)
				ax.fill_between(WAVELENGTHS, fd_avg_spectra+fd_avg_spectra_std,fd_avg_spectra-fd_avg_spectra_std,alpha=0.3, color=FD_color)
				ax.plot(WAVELENGTHS, td_avg_spectra, color=TD_color)
				ax.fill_between(WAVELENGTHS, td_avg_spectra+td_avg_spectra_std,td_avg_spectra-td_avg_spectra_std,alpha=0.3, color=TD_color)
				ax.legend(['HF Device', '+/-', 'FD-Only', '+/-', 'TD-Only', '+/-'])
				ax.set_xlabel('Wavelength (nm)')
				ax.set_ylabel('Intensity (counts)')
				ax.set_title('Comparison of Spectra at Constant Power (500'r'$\pm$'+'100 mW/cm2)')

				plt.savefig(local_deconv_save_path+"Hyperfluorescence_Spectral_Comparison.png")
				plt.clf()
				#Section finished

				POWER_BIN_SIZE = 100
				num_power_bins = 0
				cutoff_power = min (max_fd_power, max_td_power, max_hf_power)
				num_power_bins = math.ceil(cutoff_power / POWER_BIN_SIZE) #Number of bins

				x = np.arange(POWER_BIN_SIZE/2, cutoff_power+POWER_BIN_SIZE/2, POWER_BIN_SIZE)
				fig, ax = plt.subplots()
				boost_summary = []
				hfqy_summary = []

				BOOST = []
				HFQY = []
				for pixel in OLED_PIXEL_NAMES:
					try:
						hf_index = next(i for i, e in enumerate(hf_working_pixels) if pixel in e)
					except:
						hf_index = -1
					try:
						fd_index = next(i for i, e in enumerate(fd_working_pixels) if pixel in e)
					except:
						fd_index = -1
					try:
						td_index = next(i for i, e in enumerate(td_working_pixels) if pixel in e)
					except:
						td_index = -1
					if hf_index > 0 and td_index > 0 and fd_index > 0: #Pixels found in all three OLEDs
						#print("Shared Pixel: ", pixel)
						boost = []
						hfqy = []
						for i in range (0, num_power_bins):
							hf_rows = hf_cutoff_data.loc[(hf_cutoff_data['Power Density (mW/cm2)'] >= POWER_BIN_SIZE*i) & (hf_cutoff_data['Power Density (mW/cm2)'] < POWER_BIN_SIZE*(i+1)) & (hf_cutoff_data['Pixel'].str.contains(pixel))]
							fd_rows = fd_cutoff_data.loc[(fd_cutoff_data['Power Density (mW/cm2)'] >= POWER_BIN_SIZE*i) & (fd_cutoff_data['Power Density (mW/cm2)'] < POWER_BIN_SIZE*(i+1)) & (fd_cutoff_data['Pixel'].str.contains(pixel))]
							td_rows = td_cutoff_data.loc[(td_cutoff_data['Power Density (mW/cm2)'] >= POWER_BIN_SIZE*i) & (td_cutoff_data['Power Density (mW/cm2)'] < POWER_BIN_SIZE*(i+1)) & (td_cutoff_data['Pixel'].str.contains(pixel))]
							within_average_region = False
							if x[i] < POWER_THRESHOLD+POWER_PLUSMINUS and x[i] > POWER_THRESHOLD-POWER_PLUSMINUS:
								within_average_region = True

							if len(hf_rows) > 0 and len(td_rows) > 0 and len(fd_rows) > 0:

								hf_fd = np.average(hf_rows['FD_BETA'])
								hf_td = np.average(hf_rows['TD_BETA'])
								fd = np.average(fd_rows['FD_TOT'])
								td = np.average(td_rows['TD_TOT'])

								boost . append ( ( hf_fd - fd ) / fd * 100 )#How much FD do we gain?
								hfqy . append ( (hf_fd) / (td - hf_td) * 100 ) #What % of TD light lost becomes FD

								if within_average_region:
									boost_summary.append(( hf_fd - fd ) / fd * 100)
									hfqy_summary.append((hf_fd) / (td - hf_td) *100)
							else:
								boost.append(np.NaN)
								hfqy.append(np.NaN)

						BOOST.append(boost)
						HFQY.append(hfqy)
						ax.scatter(x, boost, color='green')
						ax.scatter(x, hfqy, color='red')
				plt.ylim([None, 200])
				ax.set_xlabel('Power Density (mW/cm2)')
				ax.set_ylabel('Percent (%)')
				ax.legend(['FD Boost', 'TD Conversion'])
				ax.set_title('Comparison of Hyperfluorescent Device to TD/FD-Only Controls (Ratios)')
				plt.savefig(local_deconv_save_path+HF_device_id+'_BOOST_HFQY.png')

				average_boost = str(np.median(boost_summary))[0:5]
				average_hfqy = str(np.median(hfqy_summary))[0:5]

				print("Average Increase in FD Emission for HF Device Compared to FD-Only Control: " + average_boost + " %")
				print("Average Conversion of TD Emission in TD-Only Device to FD Emission in HF Device: " + average_hfqy + " %")
				print()

			input_row ['Status'] = 'Complete'
			input_row ['FD (%) @  L = 250 cd/m2'] = str(threshold_fdpercent*100)[0:5]
			input_row ['Fit R^2 @ L = 250 cd/m2'] = str(threshold_r2)[0:5]
			input_row ['FD Intensity (counts) @ P=500 mW/cm2'] = format_e(threshold_fd_total)

			if full_HFQY_calculation:
				input_row ['FD Boost (%) @ P=500 mW/cm2'] = average_boost
				input_row ['TD Conversion (%) @ P=500 mW/cm2'] = average_hfqy
			else:
				input_row ['FD Boost (%) @ P=500 mW/cm2'] = "N/A"
				input_row ['TD Conversion (%) @ P=500 mW/cm2'] = "N/A"
			rows_to_edit.append(input_index)
	
	#print(rows_to_edit)
	#upload = [spectral_input_df.columns.values.tolist()] + spectral_input_df.values.tolist()
	convert_df_to_xlsx(MASTER_OLED_FILE, OLED_DECONV_INPUT_SHEET, OLED_DECONV_INPUT_COLUMNS, spectral_input_df, rows_to_edit, False)
def format_e(n):
    a = '%E' % n
    return (a.split('E')[0].rstrip('0').rstrip('.'))[0:4] + 'E' + a.split('E')[1]

#This looks for simple correlations... how does changing one thickness affect output?
def check_correlations(working_devices,device_id, device_output_folder):

	X_df = pd.read_csv(device_output_folder.replace("Correlations", "Raw Data")+device_id+'_stack_details.csv', delimiter='\t')
	X_matrix = np.array(X_df.iloc[:, 2:].values).astype(float)
	material_list = X_df.columns.values[2:]
	#print ("X_matrix shape: ", X_matrix.shape, " (Should be 20xNumber of Layers)")
	#print (material_list)

	#Step 2: Get Y matrix from the oled output data!
	Y = np.array(working_devices.iloc[:, 1:].values).astype(float)
	#print ("Y_matrix shape: ", Y.shape, " (Should be number of working pixels x number of outputs)")

	X = []
	for index, row in working_devices.iterrows():
	    if row['Pixel'] in X_df['Pixel'].values:
	        X.append(list(X_matrix[index]))
	X = np.array(X).astype(float)
	#print("Updated X_matrix shape: ", X.shape, " (Should be number of working pixels x number of inputs)")

	correlation_df = pd.DataFrame()
	correlation_df['Material']=material_list

	desired_outputs = ['FD_ct', 'FD_%']
	y_count = 0
	for y in np.transpose(Y):
	    y_name = desired_outputs[y_count]
	    material_count = 0
	    rs = []

	    for material in material_list:
	        x = X[:, material_count]
	        r = np.corrcoef(x, y)[0,1]
	        rs.append(r)
	        #print("Material: " + material + ", Correlation: " + str(r)[0:5])
	        material_count+=1

	    correlation_df[y_name]=rs
	    best_correlation_index = np.argmax(np.abs(rs))
	    best_material = material_list[best_correlation_index]
	    #print("Material most correlated to " + y_name + ": " + best_material + ", r=" + str(rs[best_correlation_index])[0:5])
	    
	    fig, ax = plt.subplots()
	    plt.scatter(X[:, best_correlation_index], y)
	    plt.title("Top Correlation for: " + y_name + " r = " + str(rs[best_correlation_index])[0:5])
	    ax.set_xlabel(best_material + " Thickness (" + r'$\AA$' + ')')
	    ax.set_ylabel(y_name)
	    plt.savefig(device_output_folder+"/"+y_name.split('(')[0]+'_top_correlation_deconv.png')
	    plt.clf()

	    y_count+=1
	correlation_df.to_csv(device_output_folder+"/"+device_id+"_correlations_deconv.csv", sep='\t', encoding='utf-8')

	return None

def create_visual_report(lum_oled_data, pow_oled_data, run_id, local_file_save_path, color_in):
    RECT_WIDTH = 0.5
    RECT_HEIGHT = 0.5
    summary_data = pd.DataFrame(columns=['Pixel','FD_ct', 'FD_%'])

    print("Creating visual report...")
    fig, ax = plt.subplots()
    count=0
    for i in range (0, 4):
        for j in range (0, 5):
            pixel_name = OLED_PIXEL_NAMES[count] #A1 all the way to D5
            lum_pixel_data = lum_oled_data[lum_oled_data['Pixel'].str.contains(pixel_name)]
            pow_pixel_data = pow_oled_data[pow_oled_data['Pixel'].str.contains(pixel_name)]
            #print(pixel_data)

            if len(lum_pixel_data) > 0:
                fd_avg = np.average(pow_pixel_data['FD_BETA'].values)
                fd_pt_avg =  np.average(lum_pixel_data['FD_PERCENT'].values)

                fd_string = "Intensity:\n" + format_e(fd_avg)+ ' counts'
                fd_pt_string = "%FD@250cd/m2:\n" + str(fd_pt_avg*100)[0:5] + " %"

                color = color_in
                full_string = fd_string + '\n' + fd_pt_string
                color_alpha = fd_avg/max(pow_oled_data['FD_BETA'])

                summary_data.loc[len(summary_data)]=[pixel_name, fd_avg, fd_pt_avg]

                ax.text((i+0.5)*RECT_WIDTH,RECT_HEIGHT*5-(j+0.5)*RECT_HEIGHT, full_string,fontsize="x-small", color="black", 
                    horizontalalignment='center', verticalalignment='center')
            else:
                ax.text((i+0.5)*RECT_WIDTH,RECT_HEIGHT*5-(j+0.5)*RECT_HEIGHT, "No Data",fontsize="x-small", color="black", 
                    horizontalalignment='center', verticalalignment='center')
                color = '#666666'
                color_alpha = 1

            ax.text(i*RECT_WIDTH+0.02,RECT_HEIGHT*5-j*RECT_HEIGHT-0.10, pixel_name,fontsize="x-small", color="black")
            ax.add_patch(Rectangle((i*RECT_WIDTH,RECT_HEIGHT*4-j*RECT_HEIGHT), RECT_WIDTH, RECT_HEIGHT, facecolor=color, alpha=color_alpha, edgecolor='black'))
            count+=1
    ax.set_ylim(0,RECT_HEIGHT*5)
    ax.set_xlim(0,RECT_WIDTH*4)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(run_id + " Visual Report - HF Deconvolution")
    plt.savefig(local_file_save_path+'/'+run_id+'_visual_deconvoluted.png')             
    return summary_data

def get_offline_spreadsheet_as_df(spreadsheet_file, range_name, range_cols):
    return pd.read_excel(spreadsheet_file, sheet_name=range_name, usecols=range_cols)

#This is to edit excel with the new data in rows_to_edit
def convert_df_to_xlsx(spreadsheet_file, range_name, range_cols, dataframe, rows_to_edit, add_rows):
    workbook = load_workbook(spreadsheet_file)
    active_sheet = workbook[range_name]
    
    #Should probably do something more robust
    for i in rows_to_edit:
        if add_rows:
            active_sheet.insert_rows(idx=i+2)
        row = dataframe.iloc[i].values
        for j in range (0, len(row)):
            cell = ascii_uppercase[j] + str(i+2) 
            active_sheet[cell]=row[j]
    workbook.save(filename=spreadsheet_file)

#These methods are out of date... refer to online spreadsheet
def getService(pathName, credentialsName, scopes, a1, a2):
    creds = None
    if os.path.exists(pathName):
        with open(pathName, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentialsName, SCOPES_DRIVE)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(pathName, 'wb') as token:
            pickle.dump(creds, token)
    return build(a1, a2, credentials=creds)
def get_spreadsheet_results(service, spreadsheet_id, range_name):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,range=range_name).execute()
    return result.get('values', [])
def update_spreadsheet(service, spreadsheet_id, range_name, body):
    value_input_option='USER_ENTERED'
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_name,
        valueInputOption=value_input_option, body=body).execute()
def get_dataframe(service, file_id, file_range):
    data = get_spreadsheet_results(service, file_id, file_range)
    return pd.DataFrame(data[1:], columns=data[0])
if __name__ == '__main__':
    main()
