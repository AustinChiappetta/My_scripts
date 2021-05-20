#!/usr/bin/env python3
import sys
import subprocess
import numpy as np
import pandas as pd
# from scipy.spatial import distance
#import scipy

#print(pd.__version__) #requires 1.2.4

#input should be the absolute or relative path to a folder containing two subfolders (labeled as desired, e.g., "gfp" and "mcherry")
#subfolders should each contain the processed_data.txt files generated from batch_processing.py for the two channels 
	#for which colocalization is to be calculated

#Final version of program should be ask for x, y, and z distances, and max separation threshold

#iterate through subfolders containing files for fields

#make list of subfolders to iterate through.
# def subfolder_getter(folder_path_arg, Windows):
def subfolder_getter(folder_path_arg):
#	print('start')
	folder_path = folder_path_arg
	# if Windows:
	# 	folder_path = os.path.normpath(folder_path)
	subs = subprocess.run(["ls", "-v", folder_path], stdout=subprocess.PIPE, text=True) #use bash command to get sorted subdirectories
#	print('end')
	subs = subs.stdout
#	print(subs)
#	print()

	#print(type(subs))
	temp_subfolders = subs.splitlines()
#	print('order of thresholds from initial getter is:', subfolders_thresholds)
#	print()
#	subfolders_thresholds = []
	subfolders_list = []
	for sub in temp_subfolders:
#		print('sub is', sub)
		if '.' in sub: #only grabs subdirectories, ignores files
#			print('skipped')
			continue
#		subfolders_thresholds += [sub]
		# if Windows:
		# 	sf = folder_path_arg + sub + '\\'
		# else:
		# 	sf = folder_path_arg + sub + '/'

		sf = folder_path_arg + sub + '/'
#		print('sf path is', sf)
		subfolders_list += [sf]
#	print()
#	print(subfolders_list, subfolders_thresholds)
#	return(subfolders_list, subfolders_thresholds)
	return(subfolders_list)

#get name of processed_data.txt files 
# def data_filename_getter(subfolder_path, Windows):
def data_filename_getter(subfolder_path):
#	print('getting loc filename. full path is:', subfolder_path)
	arg = subfolder_path + '*processed_data.txt'
	# if Windows:
	# 	arg = os.path.normpath(arg)
#	print('path of loc file is:', arg)
	data_filename = subprocess.run(["ls", arg], stdout=subprocess.PIPE, text=True)
	data_filename = data_filename.stdout.strip()
	return(data_filename)


# def data_handler(sub_list, subfolder_path, xpixel_dist, ypixel_dist, zpixel_dist, Windows):
def data_handler(sub_list, subfolder_path, xpixel_dist, ypixel_dist, zpixel_dist, threshold_dist):
	# num_spots_data = []
	# num_molecules_data = []
	# thresholds_used = []

	#sub = sub_list[0] #grab first item in list for testing this block
	first_sub = sub_list[0]
	# if Windows:
	# 	first_sub_name = first_sub.split('\\')[-2]
	# else:
	# 	first_sub_name = first_sub.split('/')[-2]
	first_sub_name = first_sub.split('/')[-2]

	second_sub = sub_list[1]
	# if Windows:
	# 	second_sub_name = second_sub.split('\\')[-2]
	# else:
	# 	second_sub_name = second_sub.split('/')[-2]
	second_sub_name = second_sub.split('/')[-2]

	# first_data_filename = data_filename_getter(first_sub, Windows)
	first_data_filename = data_filename_getter(first_sub)
	# second_data_filename = data_filename_getter(second_sub, Windows)
	second_data_filename = data_filename_getter(second_sub)

	first_df = make_dataframe(first_data_filename, xpixel_dist, ypixel_dist, zpixel_dist)
	second_df = make_dataframe(second_data_filename, xpixel_dist, ypixel_dist, zpixel_dist)

	# colocalization(first_df, second_df, first_sub_name, second_sub_name, subfolder_path, xpixel_dist, ypixel_dist, zpixel_dist, Windows)
	coloc_df = colocalization(first_df, second_df, first_sub_name, second_sub_name, subfolder_path, xpixel_dist, ypixel_dist, zpixel_dist, threshold_dist)

	summary_plotting(first_df, second_df, first_sub_name, second_sub_name, coloc_df, subfolder_path)

#	print()
	#return(num_spots_data, num_molecules_data, thresholds_used)


#open processed_data.txt files corresponding to the two channels for the field and make pandas dataframes

#def make_dataframe(data_filename, xres, yres, zres):
def make_dataframe(data_filename, xpixel_dist, ypixel_dist, zpixel_dist):
#	print()
#	print('reading filename:', data_filename)
	df = pd.read_csv(data_filename, sep='\t')
	sorted_df = df.sort_values(by=['z'], ascending=False)
	#print('before pixel conversion')
	#print(sorted_df.head())

	# xpixel_dist = 102.4 #in nanometers
	# ypixel_dist = 102.4
	# zpixel_dist = 250 #zstep

	#convert pixel coordinates to nm.
	sorted_df['x'] = xpixel_dist * sorted_df['x']
	sorted_df['y'] = ypixel_dist * sorted_df['y']
	sorted_df['z'] = zpixel_dist * sorted_df['z']
	

	#make a column of coordinate vectors [x,y,z] to be used for making a distance matrix
	#sorted_df['coordinates'] = (sorted_df['x'], sorted_df['y'], sorted_df['z'])

	coords = []
	for x, y, z in zip(sorted_df['x'], sorted_df['y'], sorted_df['z']):
		coord = (str(x) + '_' + str(y) + '_' + str(z))
		#coord = (x, y, z)
		coords += [(coord)]

	sorted_df['coordinates'] = coords
	# print(coords[:10])
	
	# print(sorted_df.head())
	# print(sorted_df.columns.tolist())

	return(sorted_df)

# 	fs = open(data_filename, 'r')
# 	count = 0
# 	#initialize data structures
# 	x_data = []
# 	y_data = []
# 	z_data = []
# 	intensity_data = []
# 	scaled_data []
# 	molecules_data = []

# 	#iterate through file
# 	for line in fs:
# 		if 'inensity' in line:
# 			continue #skip headers

# 		#if count > 5:
# 		#	break
# #		print(line)
# #		print(type(line))
# #		print()
# 		#fields = line.splitlines()
# 		#print(fields)
# 		fields = line.strip()
# 		fields = line.split()
# 		#print()
# 		#print(fields)
		
# 		x_coord = float(fields[0])
# 		y_coord = float(fields[1])
# 		z_coord = float(fields[2])
# 		intensity = float(fields[3])
# 		scaled = float(fields[4])
# 		quantified_molecules = int(fields[5])

# 		y_data += [y_coord]
# 		x_data += [x_coord]
# 		z_data += [z_coord]
# 		intensity_data += [intensity]
# 		#count += 1
# 	fs.close()

# def colocalization(df1, df2, df1_spot_name, df2_spot_name, outputpath, xpixel_dist, ypixel_dist, zpixel_dist, Windows):
def colocalization(df1, df2, df1_spot_name, df2_spot_name, outputpath, xpixel_dist, ypixel_dist, zpixel_dist, threshold_dist):
	threshold = threshold_dist
	# threshold = 500 #in nanometers

	outputpathname = outputpath + df1_spot_name + '_' + df2_spot_name + '_' + 'colocalized.tsv'

	summaryoutputname = outputpath + df1_spot_name + '_' + df2_spot_name + '_' + 'colocalized_summary.txt'

	# if Windows:
	# 	outputpathname = os.path.normpath(outputpathname)
	
	df1_columns = df1.columns.tolist()
	df2_columns = df2.columns.tolist()

	pixel_cols_list = ['x_pixel', 'y_pixel', 'z_pixel']

	spot1_columns_list = []
	for pixel_col in pixel_cols_list:
		spot1_columns_list += [df1_spot_name + '_' + pixel_col]
	for col in df1_columns[:-1]:
		spot1_columns_list += [df1_spot_name + '_' + col]


	spot2_columns_list = []
	for pixel_col in pixel_cols_list:
		spot2_columns_list += [df2_spot_name + '_' + pixel_col]
	for col in df2_columns[:-1]:
		spot2_columns_list += [df2_spot_name + '_' + col]

	combined_spot_columns_list = spot1_columns_list + spot2_columns_list + ['distance (nm)']

	# print(combined_spot_columns_list)

	df1_coords = df1[['x', 'y', 'z']].to_numpy()

	# print()
	# print(df1_coords)
	# print(np.shape(df1_coords))
	# print()
	
	df2_coords = df2[['x', 'y', 'z']].to_numpy()
	# print(np.shape(df2_coords))
	# print()

	spot1_coord_list = df1['coordinates'].tolist()
	spot2_coord_list = df2['coordinates'].tolist()

	#columnlist = sorted_df.columns.tolist()

	# distances = distance_matrix(df1_coords, df2_coords)
	distances = np.linalg.norm(df1_coords[:, None, :] - df2_coords[None, :, :], axis=-1)
	
	putative_array = np.where(distances < threshold, distances, np.nan) #enforce maximum threshold

	putative_df = pd.DataFrame(putative_array, index = spot1_coord_list, columns = spot2_coord_list)
	# print(putative_df.head())
	# putative_df = putative_df.dropna(axis = 0, how='all')
	# putative_df = putative_df.dropna(axis = 1, how='all')

	# print('after dropping rows/cols with all nan')
	# print(putative_df.head())
	# print('shape is', np.shape(putative_df))

	# print(distances)
	# print(np.shape(distances))
	# print()
	number_nans = putative_df.isnull().sum().sum()
	# print('number of nans is', number_nans)
	# print()

	# print(np.amin(putative_df))
	# print(np.nanmax(putative_df))
	# print(np.nanmin(putative_df))
	# print()

	putative_shape = putative_df.shape
	# print('putative shape is', putative_shape)

	total_number_vals = (int(putative_shape[0]) * int(putative_shape[1]))
	# print('total number of values is', total_number_vals)

	lowest_idx = np.nanargmin(putative_df)
	# print('flattened lowest idx is', lowest_idx)

	unraveled = np.unravel_index(lowest_idx, putative_shape) #returns index (rows, cols)
	# print('unraveled is', unraveled)
	# print('lowest is', putative_df.iloc[unraveled])

	# spot1_coord = spot1_coord_list[unraveled[0]]
	# spot2_coord = spot2_coord_list[unraveled[1]]

	# print('\nspot1 coord', spot1_coord, '\nspot2 coord', spot2_coord)

	# print(df1.iloc[unraveled[0]]['x']) #testing functionality of iloc in relation to my dfs
	# print(df2.iloc[unraveled[1]])

	# print()
	# for vals in spot2_row:
	# 	print(vals)

	#initialize lists

	spot1_x_list = []
	spot1_y_list = []
	spot1_z_list = []
	spot1_intensity_list = []
	spot1_scaled_intensity_list = []
	spot1_number_mols_quant_list = []

	spot2_x_list = []
	spot2_y_list = []
	spot2_z_list = []
	spot2_intensity_list = []
	spot2_scaled_intensity_list = []
	spot2_number_mols_quant_list = []

	distance_list = []

	number_nans = putative_df.isnull().sum().sum()

	while number_nans < total_number_vals:
		lowest_idx = np.nanargmin(putative_df) #find index of smallest distance (is flattened)
		#print('flattened lowest idx is', lowest_idx)
		unraveled = np.unravel_index(lowest_idx, putative_shape) #get the index of lowest in native df (putative df)

		#for increased versatility, future version should change this list-making to making a dictionary

		spot1_x = df1.iloc[unraveled[0]]['x']
		spot1_y = df1.iloc[unraveled[0]]['y']
		spot1_z = df1.iloc[unraveled[0]]['z']
		spot1_intensity = df1.iloc[unraveled[0]]['intensity']
		spot1_scaled_intensity = df1.iloc[unraveled[0]]['scaled intensity']
		spot1_number_mols_quant = df1.iloc[unraveled[0]]['number of molecules quantified']

		spot2_x = df2.iloc[unraveled[1]]['x']
		spot2_y = df2.iloc[unraveled[1]]['y']
		spot2_z = df2.iloc[unraveled[1]]['z']
		spot2_intensity = df2.iloc[unraveled[1]]['intensity']
		spot2_scaled_intensity = df2.iloc[unraveled[1]]['scaled intensity']
		spot2_number_mols_quant = df2.iloc[unraveled[1]]['number of molecules quantified']

		distance = putative_df.iloc[unraveled]

		#as I loop through, add each value to new lists

		spot1_x_list += [spot1_x]
		spot1_y_list += [spot1_y]
		spot1_z_list += [spot1_z]
		spot1_intensity_list += [spot1_intensity]
		spot1_scaled_intensity_list += [spot1_scaled_intensity]
		spot1_number_mols_quant_list += [spot1_number_mols_quant]

		spot2_x_list += [spot2_x]
		spot2_y_list += [spot2_y]
		spot2_z_list += [spot2_z]
		spot2_intensity_list += [spot2_intensity]
		spot2_scaled_intensity_list += [spot2_scaled_intensity]
		spot2_number_mols_quant_list += [spot2_number_mols_quant]

		distance_list += [distance]

		#make all values in row and column nans to prevent double counting of other spots with this pair of spots
		# print('site of error')
		# print()
		# print(putative_df.iloc[unraveled[0]])
		putative_df.iloc[unraveled[0]] = np.nan
		#df.iloc[1] = 'diff' #change all in row
		putative_df.iloc[:, [unraveled[1]]] = np.nan
		#df.iloc[:, 1] = 'diff' #change all in column
		number_nans = putative_df.isnull().sum().sum()
	
	#after loop is done, use lists to make colocalization df

	#convert distance coordinates back to pixel coordinates 
	spot1_xpixel_list = [x/xpixel_dist for x in spot1_x_list]
	spot1_ypixel_list = [y/ypixel_dist for y in spot1_y_list]
	spot1_zpixel_list = [z/zpixel_dist for z in spot1_z_list]

	spot2_xpixel_list = [x/xpixel_dist for x in spot2_x_list]
	spot2_ypixel_list = [y/ypixel_dist for y in spot2_y_list]
	spot2_zpixel_list = [z/zpixel_dist for z in spot2_z_list]


	#df = pd.DataFrame([arr1, arr2])
	data_list = [spot1_xpixel_list, spot1_ypixel_list, spot1_zpixel_list, spot1_x_list, spot1_y_list, spot1_z_list, 
		spot1_intensity_list, spot1_scaled_intensity_list, spot1_number_mols_quant_list, spot2_xpixel_list, spot2_ypixel_list,
		spot2_zpixel_list, spot2_x_list, spot2_y_list, spot2_z_list, spot2_intensity_list, spot2_scaled_intensity_list, 
		spot2_number_mols_quant_list, distance_list]

	colocalized_df = pd.DataFrame() #initialize new dataframe

	#df = pd.DataFrame([arr1, arr2])

	for count, d in enumerate(data_list):
		colocalized_df[combined_spot_columns_list[count]] = d #add new columns from data lists

	colocalized_df.to_csv(path_or_buf = outputpathname, index=False, sep='\t')

	df1_num_spots = str(df1.shape[0])
	df2_num_spots = str(df2.shape[0])
	num_colocalized = str(len(distance_list))

	fs = open(summaryoutputname, 'w')
	# outputpath + df1_spot_name + '_' + df2_spot_name + '_' + 'colocalized_summary.txt'
	fs.write('Total number of ' + df1_spot_name + ' spots: ' + df1_num_spots + '\n')
	fs.write('Total number of ' + df2_spot_name + ' spots: ' + df2_num_spots + '\n')
	fs.write('Number of colocalized spots: ' + num_colocalized)
	fs.close()


	print()
	print('Dataframe of colocalized spots has been saved as a tab-separated file:', outputpathname)

	return(colocalized_df)

	#go through the matrix in the order of smallest values to largest values
	#use nanargmin to find the index of the smallest value
	#save value to new df
	#change all other putative pairwise distances to each of those spots to nans (avoid double counting)
	#stop when putative df contains only nan


#calculate the Euclidian distance between each pair of green spots and red spots.
#distance = ((dx**2) + (dy**2) + (dz**2))**(0.5)



#plot a pie chart of the number of colocalized spots and non-colocalized spots. (or bar plots?)
#plot a histogram of signal intentensities for all spots, and plot a histogram of signal intensities for colocalized spots (overlapping)
#2 by 2 plot: GFP pie chart then GFP histos over MCHERRY pie chart and MCHERRY histos
#re-normalize the colocalized signal intensities?
#Save the plots.

def summary_plotting(first_df, second_df, first_sub_name, second_sub_name, coloc_df, folder_path):
	import matplotlib.pyplot as plt
	df1 = first_df
	df2 = second_df
	coloc_df = coloc_df 
	
	df1_num_spots = df1.shape[0]
	# print('df1_num_spots is', df1_num_spots)
	df2_num_spots = df2.shape[0]
	# print('df2_num_spots is', df2_num_spots)
	num_colocalized = coloc_df.shape[0]
	# print('num colocalized is', num_colocalized)

	df1_num_noncoloc = df1_num_spots - num_colocalized
	df2_num_noncoloc = df2_num_spots - num_colocalized

	# spot1_all_molecules_quant = df1['number of molecules quantified']
	# spot1_coloc_molecules_quant = coloc_df[first_sub_name + '_number of molecules quantified']

	# spot2_all_molecules_quant = df2['number of molecules quantified']
	# spot2_coloc_molecules_quant = coloc_df[second_sub_name + '_number of molecules quantified']

	# pie_labels = ['Non-colocalized', 'Colocalized']

	# # bins = range(0, (max(x) + 1), 1)

	# fig, ax = plt.subplots(2,2, figsize=(10,6))
	# ax[0,0].pie([df1_num_noncoloc, num_colocalized], labels=pie_labels, autopct='%1.1f%%', startangle=90)
	# ax[0,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	# ax[0,0].title.set_text('Portion of ' + first_sub_name + ' spots colocalized')

	# ax[0,1].hist(spot1_all_molecules_quant, bins = range(0, (int(max(spot1_all_molecules_quant)) + 1), 1), alpha=0.5, label='All spots')
	# ax[0,1].hist(spot1_coloc_molecules_quant, bins = range(0, (int(max(spot1_all_molecules_quant)) + 1), 1), alpha=0.5, label='Colocalized spots')
	# ax[0,1].set_xlabel('Number of molecules per spot')
	# ax[0,1].set_ylabel('Counts')
	# ax[0,1].legend()
	# ax[0,1].title.set_text('Number of molecules in colocalized vs all ' + first_sub_name + ' spots')

	# ax[1,0].pie([df2_num_noncoloc, num_colocalized], labels=pie_labels, autopct='%1.1f%%', startangle=90)
	# ax[1,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	# ax[1,0].title.set_text('Portion of ' + second_sub_name + ' spots colocalized')

	# ax[1,1].hist(spot2_all_molecules_quant, bins = range(0, (int(max(spot2_all_molecules_quant)) + 1), 1), alpha=0.5, label='All spots')
	# ax[1,1].hist(spot2_coloc_molecules_quant, bins = range(0, (int(max(spot2_all_molecules_quant)) + 1), 1), alpha=0.5, label='Colocalized spots')
	# ax[1,1].set_xlabel('Number of molecules per spot')
	# ax[1,1].set_ylabel('Counts')
	# ax[1,1].legend()
	# ax[1,1].title.set_text('Number of molecules in colocalized vs all ' + second_sub_name + ' spots')




	spot1_all_spot_intensities = df1['scaled intensity']
	spot1_coloc_intensities = coloc_df[first_sub_name + '_scaled intensity']

	spot2_all_spot_intensities = df2['scaled intensity']
	spot2_coloc_intensities = coloc_df[second_sub_name + '_scaled intensity']

	pie_labels = ['Non-colocalized', 'Colocalized']

	bins = np.linspace(0, 300, num=20)

	fig, ax = plt.subplots(2,2, figsize=(10,6))
	ax[0,0].pie([df1_num_noncoloc, num_colocalized], labels=pie_labels, autopct='%1.1f%%', startangle=90)
	ax[0,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	ax[0,0].title.set_text('Portion of ' + first_sub_name + ' spots colocalized')

	ax[0,1].hist(spot1_all_spot_intensities, bins = np.linspace(0, max(spot1_all_spot_intensities), num=40), alpha=0.5, label='intensities of all spots')
	ax[0,1].hist(spot1_coloc_intensities, bins = np.linspace(0, max(spot1_all_spot_intensities), num=40), alpha=0.5, label='intensities of colocalized spots')
	ax[0,1].set_xlabel('Spot intensity (arbitrary units)')
	ax[0,1].set_ylabel('Counts')
	ax[0,1].legend()
	ax[0,1].title.set_text('Intensities of colocalized vs all ' + first_sub_name + ' spots')

	ax[1,0].pie([df2_num_noncoloc, num_colocalized], labels=pie_labels, autopct='%1.1f%%', startangle=90)
	ax[1,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	ax[1,0].title.set_text('Portion of ' + second_sub_name + ' spots colocalized')

	ax[1,1].hist(spot2_all_spot_intensities, bins = np.linspace(0, max(spot2_all_spot_intensities), num=40), alpha=0.5, label='intensities of all spots')
	ax[1,1].hist(spot2_coloc_intensities, bins = np.linspace(0, max(spot1_all_spot_intensities), num=40), alpha=0.5, label='intensities of colocalized spots')
	ax[1,1].set_xlabel('Spot intensity (arbitrary units)')
	ax[1,1].set_ylabel('Counts')
	ax[1,1].legend()
	ax[1,1].title.set_text('Intensities of colocalized vs all ' + second_sub_name + ' spots')

	fig.suptitle('Summary of ' + first_sub_name + ' and ' + second_sub_name + ' colocalization')
	fig.tight_layout()
	fig_filename = folder_path + 'colocalization_summary_plot.png'
	plt.savefig(fig_filename)
	plt.show()
	print()
	print('Plot has been saved as:', fig_filename)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: python colocalization_calculator.py /PATH/TO/FOLDER/CONTAINING/DATA/SUBFOLDERS/ xpixel_to_nm_conversion ypixel_to_nm_conversion zstep max_distance_threshold')
	else:
		arg = sys.argv[1]
#		print('The argument you passed is:', arg)
#		print()
		folder_path_arg = arg
		# print('path received is:', folder_path_arg)
		# print()

		# if "\\" in folder_path_arg: #compatibility with Windows powershell. #gave up on getting this to work, ended up abondoning scipy and using numpy for generating the distance matrix
		# 	import os
		# 	Windows = True

		xpixel_dist = float(sys.argv[2]) #in nanometers
		ypixel_dist = float(sys.argv[3])
		zpixel_dist = float(sys.argv[4]) #zstep also in nm
		threshold_dist = float(sys.argv[5]) #maximum distance for defining spots as colocalized, also in nm
			
	
		if folder_path_arg[-1] != '/':
			folder_path_arg = folder_path_arg + '/' #the algorithm requires a trailing '/' at the end of the pathname
		# subs = subfolder_getter(folder_path_arg, Windows)
		subs = subfolder_getter(folder_path_arg)

		#used these pixel distances for analysis of images taken with 63x lens when I was writing the algorithm
		# xpixel_dist = 102.4 #in nanometers
		# ypixel_dist = 102.4
		# zpixel_dist = 250 #zstep

		# data_handler(subs, folder_path_arg, xpixel_dist, ypixel_dist, zpixel_dist, Windows)
		data_handler(subs, folder_path_arg, xpixel_dist, ypixel_dist, zpixel_dist, threshold_dist)

		#spots_data, molecules_data, thresholds = folder_iterator(subs)
#		print(thresholds_floats, spots_data)
#		threshold_spots_plotter(thresholds_floats, spots_data, folder_path_arg)
#		make_summary_file(folder_path_arg, subs, spots_data, molecules_data, thresholds)
		print()
		print('Algorithm completed')