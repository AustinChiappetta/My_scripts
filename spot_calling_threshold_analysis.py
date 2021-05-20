#!/usr/bin/env python3
import sys
import subprocess
import numpy as np

#make list of subfolders to iterate through. Save threshold values
def subfolder_getter(folder_path_arg):
#	print('start')
	folder_path = folder_path_arg
	subs = subprocess.run(["ls", "-v", folder_path], stdout=subprocess.PIPE, text=True) #use bash command to get sorted subdirectories
#	print('end')
	subs = subs.stdout
#	print(subs)
#	print()

	#print(type(subs))
	temp_subfolders_thresholds = subs.splitlines()
#	print('order of thresholds from initial getter is:', subfolders_thresholds)
#	print()
	subfolders_thresholds = []
	subfolders_list = []
	for sub in temp_subfolders_thresholds:
#		print('sub is', sub)
		if '.' in sub:
#			print('skipped')
			continue
		subfolders_thresholds += [sub]
		sf = folder_path_arg + sub + '/'
#		print('sf path is', sf)
		subfolders_list += [sf]
#	print()
#	print(subfolders_list, subfolders_thresholds)
	return(subfolders_list, subfolders_thresholds)

#get name of .loc file from subfolder
def loc_filename_getter(subfolder_path):
#	print('getting loc filename. full path is:', subfolder_path)
	arg = subfolder_path + '*.loc3'
#	print('path of loc file is:', arg)
	loc_filename = subprocess.run(["ls", arg], stdout=subprocess.PIPE, text=True)
	loc_filename = loc_filename.stdout.strip()
	return(loc_filename)

#open and read .loc files

def file_reader(loc_filename):
#	print()
#	print('reading filename:', loc_filename)
	fs = open(loc_filename, 'r')
	count = 0
	intensity_data = []
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
		intensity = float(fields[3])
		intensity_data += [intensity]
		#count += 1

#	print(intensity_data[:10])

	if len(intensity_data) != 0:

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
	# print('max scaled intensity is', max(scaled))

	max_scaled = max(scaled)

	#generate bins for np.digitize
	bm = int((((max_scaled//15)*15)+15))
	# print('bm is', bm)

	bins = []
	for b in range(0, bm + 15, 15):
	    bins += [b]
	#bins += [range(0, bm, 15)]
	# print('max intensity over 300, bins is:', bins)

	#generate counts of values in each bin, analogous to manually generating a histogram to find the mean and determine x0
	bin_counts = np.digitize(scaled, bins=bins, right=True)
	# print(bin_counts)
	# print('len of bins is', len(bins))

	freq_dict = {}
	for bin in bin_counts:
		freq_dict.setdefault(bin, 0)
		freq_dict[bin] += 1

	copydict = freq_dict.copy()
	# print(copydict)
	keys = list(freq_dict.keys())
	vals = [freq_dict[x] for x in keys]
	for key in keys:
		if freq_dict[key] != max(vals):
			del copydict[key]

	dkeys = list(copydict.keys())
	if len(dkeys) > 1:
		x0 = bins[min(dkeys)]
		if x0 == 0:
			x0 = bins[min(dkeys)+1]
		# print('x0 is', x0)

	elif len(dkeys) == 1:
		# print('found smallest')
		# print('dkeys is', dkeys)
		# print('dkeys[0] is', dkeys[0])
		x0 = bins[dkeys[0]]
		# print('x0 is', x0)

# 	histo = np.histogram(scaled, bins=bins)
# #	print(histo)
# 	max = np.argmax(histo[0])
# 	x0 = histo[1][max]
#	print('x0 is:', x0)
#	print('max index:', max)
#	print('max:', histo[1][max])

	molecules = [x/x0 for x in scaled]
	# print(molecules[0])
	# molecules_rounded = [x.round() for x in molecules] #apparently x.round() doesn't work now? Was working earlier.
	molecules_rounded = [round(x) for x in molecules]
	total_molecules = np.sum(molecules_rounded)
	molecules_rounded_above_zero = []
	for m in molecules_rounded:
		if m > 0:
			molecules_rounded_above_zero += [m] #keep only spots with more than one molecule after normalization

	num_spots = len(molecules_rounded_above_zero)

	#output_file = open(loc_filename_analyzed, 'w+')
	#save threshold value and number of spots called
	return(num_spots, total_molecules)

def folder_iterator(sub_list):
	num_spots_data = []
	num_molecules_data = []

	for sub in sub_list:
		#sub = sub_list[0] #grab first item in list for testing this block
		filename = loc_filename_getter(sub)
		spots, molecules = file_reader(filename)
		num_spots_data += [spots]
		num_molecules_data += [molecules]
#	print()
	return(num_spots_data, num_molecules_data)

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


#plot number of molecules as a function of threshold
def threshold_molecules_plotter(threshold_data, molecule_data, dir):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.scatter(threshold_data, molecule_data)
        plt.xlabel('Threshold')
        plt.ylabel('Number of Molecules Detected')
        plt.title('Molecule Quantification as a Function of Threshold')
        fig_filename = dir + 'molecules_detected_plot.png'
        plt.savefig(fig_filename)
        plt.show()
        print('Plot has been saved as:', fig_filename)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: python spot_calling_thresholdhold_analysis.py ABSOLUTE/PATH/TO/FOLDER/CONTAINING/THRESHOLD/SUBFOLDERS/')
	else:
		arg = sys.argv[1]
#		print('The argument you passed is:', arg)
#		print()
		folder_path_arg = arg
		if folder_path_arg[-1] != '/':
			folder_path_arg = folder_path_arg + '/' #the algorithm requires a trailing '/' at the end of the pathname
		subs, thresholds = subfolder_getter(folder_path_arg)
#		print('subs from subfolder_getter:', subs)
#		print('thresholds from subfolder_getter:', thresholds)
#		print()
		thresholds_floats = [float(s) for s in thresholds]
		thresholds_floats = sorted(thresholds_floats)
#		print(thresholds_floats)
#		print()
#		print('thresholds floats:', thresholds_floats)
#		print('full path of subs is:', subs)
		spots_data, molecules_data = folder_iterator(subs)
#		print(thresholds_floats, spots_data)
		threshold_spots_plotter(thresholds_floats, spots_data, folder_path_arg)
		threshold_molecules_plotter(thresholds_floats, molecules_data, folder_path_arg)
		print('Algorithm completed successfully')
