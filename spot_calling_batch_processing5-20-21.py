#!/usr/bin/env python3
import sys
import subprocess
import numpy as np

#files for each field should be separated in subfolders (grab info from subfolder name)
#note: this code is modified from spot_calling_threshold_analysis.py

#make list of subfolders to iterate through.
def subfolder_getter(folder_path_arg):
#	print('start')
	folder_path = folder_path_arg
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
		sf = folder_path_arg + sub + '/'
#		print('sf path is', sf)
		subfolders_list += [sf]
#	print()
#	print(subfolders_list, subfolders_thresholds)
#	return(subfolders_list, subfolders_thresholds)
	return(subfolders_list)

#get name of .loc file 
def loc_filename_getter(subfolder_path):
#	print('getting loc filename. full path is:', subfolder_path)
	arg = subfolder_path + '*.loc3'
#	print('path of loc file is:', arg)
	loc_filename = subprocess.run(["ls", arg], stdout=subprocess.PIPE, text=True)
	loc_filename = loc_filename.stdout.strip()
	return(loc_filename)


#get name of .par file for airlocalize parameters used 
def par_filename_getter(subfolder_path):
#	print('getting par filename. full path is:', subfolder_path)
	arg = subfolder_path + '*.par3'
#	print('path of par file is:', arg)
	par_filename = subprocess.run(["ls", arg], stdout=subprocess.PIPE, text=True)
	par_filename = par_filename.stdout.strip()
	return(par_filename)

#open and read .loc files. Save info to a data structure, then write raw and processed data to a new file, 
#then calculate avg number of spots and avg number of molecules detected

def loc_file_reader(loc_filename):
#	print()
#	print('reading filename:', loc_filename)
	fs = open(loc_filename, 'r')
	count = 0
	#initialize data structures
	y_data = []
	x_data = []
	z_data = []
	intensity_data = []
	#iterate through file
	for line in fs:
		#if count > 5:
		#	break
#		print(line)
#		print(type(line))
#		print()
		#fields = line.splitlines()
		#print(fields)
		fields = line.strip()
		fields = line.split()
		#print()
		#print(fields)
		y_coord = float(fields[0])
		x_coord = float(fields[1])
		z_coord = float(fields[2])
		intensity = float(fields[3])

		y_data += [y_coord]
		x_data += [x_coord]
		z_data += [z_coord]
		intensity_data += [intensity]
		#count += 1

#	print(intensity_data[:10])

	#scale the data
	highest = str(intensity_data[0])
	highest = highest.split('.')[0] #remove decimal
	digits = int(len(highest))
	div = digits - 3
	div = 10**div
	scaled = [x/div for x in intensity_data]
#	print('len of scaled is:', len(scaled))
#	print(scaled[:10])

		#find x0
	max_scaled = max(scaled)

	step_size = 15

	#generate bins for np.digitize
	bm = int((((max_scaled//step_size)*step_size)+step_size))
	# print('bm is', bm)

	bins = []
	for b in range(0, bm + step_size, step_size):
	    bins += [b]
	#bins += [range(0, bm, 15)]
	# print('max intensity over 300, bins is:', bins)

	#generate counts of values in each bin, analogous to manually generating a histogram to find the mean and determine x0
	bin_counts = np.digitize(scaled, bins=bins, right=True)

	# #find x0
	# bins = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300])

	# #histo = np.histogram(scaled, bins=bins)
	# #print(histo)
	
	# bin_counts = np.digitize(scaled, bins=bins, right=True)
#	print(bin_counts)
#	print()

	freq_dict = {}
	for bin in bin_counts:
		freq_dict.setdefault(bin, 0)
		freq_dict[bin] += 1

	copydict = freq_dict.copy()
	keys = list(freq_dict.keys())
	vals = [freq_dict[x] for x in keys]
	for key in keys:
		if freq_dict[key] != max(vals):
			del copydict[key]

	dkeys = list(copydict.keys())
	if len(dkeys) > 1:
		x0 = bins[min(dkeys)]
	elif len(dkeys) == 1:
		x0 = bins[dkeys[0]]


	#most_freq_index = mode(bin_counts)
	#print('freq index is:', most_freq_index)
	#x0 = bins[most_freq_index]

	#max = np.argmax(histo[0])
	#x0 = histo[1][max]
#	print('x0 is:', x0)
	#print('max index:', max)
	#print('max:', histo[1][max])
#	print()

	molecules = [x/x0 for x in scaled]
	molecules_rounded = [round(x) for x in molecules]
	# molecules_rounded = [x.round() for x in molecules]
	#print('molecules rounded is:', molecules_rounded)
	total_molecules = np.sum(molecules_rounded)
	molecules_rounded_above_zero = []
	for m in molecules_rounded:
		if m > 0:
			molecules_rounded_above_zero += [m] #keep only spots with more than one molecule after normalization

	num_spots = len(molecules_rounded_above_zero)
#	print('# spots above zero:', num_spots)

	fs.close()

	#write output file
	levels = loc_filename.split('/')
	subdir_name = levels[-2]
	output_filename = loc_filename
	while output_filename[-1] != '/': #truncate path
		output_filename = output_filename[:-1]
	output_filename = output_filename + subdir_name + 'processed_data.txt'

	fs = open(output_filename, 'w+')
	#output file is tab-separated
	zipped = zip(x_data, y_data, z_data, intensity_data, scaled, molecules_rounded_above_zero)
	fs.write('x' + '\t' + 'y' + '\t' + 'z' + '\t' + 'intensity' + '\t' + 'scaled intensity' + '\t' + 'number of molecules quantified' + '\n') #add header
	#zip through x_data, y_data, z_data, intensity_data, scaled, and molecules_rounded_above_zero and write to output file
	for data in zipped:
		x = str(data[0])
		y = str(data[1])
		z = str(data[2])
		i = str(data[3])
		i_scaled = str(data[4])
		m = str(data[5])
		fs.write(x + '\t' + y + '\t' + z + '\t' + i + '\t' + i_scaled + '\t' + m + '\n')
	fs.close()

	print()
	print('Processed data saved as', output_filename)

	#save number of spots called and total molecules quantified for averages calculations later
	return(num_spots, total_molecules)

def par_file_reader(par_filename):
	#print('starting par_file_reader')
	#print('par file is:', par_filename)
	#print()
	fs = open(par_filename)
	for line in fs:
		fields = line.strip()
		fields = fields.split()
	#	print(fields)
		if 'thresh.level:' in fields:
			thresh = fields[1]
	#		print('found threshold:', thresh)
			#break
			fs.close()
			return(thresh)
	#print()
	#fs.close()

	#return(thresh)


def folder_iterator(sub_list):
	num_spots_data = []
	num_molecules_data = []
	thresholds_used = []

	for sub in sub_list:
		#sub = sub_list[0] #grab first item in list for testing this block
		loc_filename = loc_filename_getter(sub)
		par_filename = par_filename_getter(sub)
		spots, molecules = loc_file_reader(loc_filename)
		num_spots_data += [spots]
		num_molecules_data += [molecules]

		threshold = par_file_reader(par_filename)
		thresholds_used += [threshold]

#	print()
	return(num_spots_data, num_molecules_data, thresholds_used)


#Make a summary table file.
def make_summary_file(path, subs, spots, molecules, thresholds):
	avg_spots = np.mean(spots)
	avg_molecules = np.mean(molecules)

	summary_filename = path + 'summary_file.txt'
	fs = open(summary_filename, 'w+')
	zipped = zip(subs, spots, molecules, thresholds)
	fs.write('Field' + '\t' + '# Spots Detected' + '\t' + '# Molecules Quantified' + '\t' + 'Threshold Used' + '\n') #add header
	#zip and write to output file
	for data in zipped:
		sub = str(data[0])
		levels = sub.split('/')
		#print(levels)
		sub_name = levels[-2]

		spot = str(data[1])
		molecule = str(data[2])
		threshold = str(data[3])
		fs.write(sub_name + '\t' + spot + '\t' + molecule + '\t' + threshold + '\n')
	fs.write('Average' + '\t' + str(avg_spots) + '\t' + str(avg_molecules) + '\t' + 'NA')

	fs.close()

	print()
	print('Average number of spots called per field:', avg_spots)
	print('Average number of molecules quantified:', avg_molecules)
	print('Summary file saved as:', summary_filename)


#plot number of spots as a function of threshold
def threshold_spots_plotter(threshold_data, spot_data, dir):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	plt.scatter(threshold_data, spot_data)
	plt.xlabel('Threshold')
	plt.ylabel('Number of Spots Detected')
	plt.title('Spot Detection as a Function of Threshold')
	fig_filename = dir + 'spots_detected_plot.png'
	plt.savefig(fig_filename)
	plt.show()
	print('Plot has been saved as:', fig_filename)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: python spot_calling_batch_processing.py ABSOLUTE/PATH/TO/FOLDER/CONTAINING/IMAGE/SUBFOLDERS/')
	else:
		arg = sys.argv[1]
#		print('The argument you passed is:', arg)
#		print()
		folder_path_arg = arg
		if folder_path_arg[-1] != '/':
			folder_path_arg = folder_path_arg + '/' #the algorithm requires a trailing '/' at the end of the pathname
		subs = subfolder_getter(folder_path_arg)
#		print('subs from subfolder_getter:', subs)
#		print('thresholds from subfolder_getter:', thresholds)
#		print()
#		thresholds_floats = [float(s) for s in thresholds]
#		thresholds_floats = sorted(thresholds_floats)
#		print(thresholds_floats)
#		print()
#		print('thresholds floats:', thresholds_floats)
#		print('full path of subs is:', subs)
		spots_data, molecules_data, thresholds = folder_iterator(subs)
#		print(thresholds_floats, spots_data)
#		threshold_spots_plotter(thresholds_floats, spots_data, folder_path_arg)
		make_summary_file(folder_path_arg, subs, spots_data, molecules_data, thresholds)
		print()
		print('Algorithm completed successfully')
