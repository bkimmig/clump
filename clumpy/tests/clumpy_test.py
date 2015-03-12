import numpy as np 
import matplotlib.pylab as plt

# import the module
# import clumpy

# This is an example of how to use clumpy. It will create a fake data set 
# simulating an object with a coherent velocity and some field noise in 
# front of it.

# Situation:
# Assume we have a telescope that can target 400 stars. We try to get only 
# globular cluster stars in our telescope (with ~1 deg field of view) but 
# that is hard to do. We get contamination from the field making it hard to 
# decide whether the stars members of the cluster or the field. 
# Below I will simulate some data and use the EM algorithm to find likely 
# members of the cluster. 

def rv_plot (rv_inst, fig_name='rad_vel', show=True):
	fig = plt.figure(fig_name)
	ax1 = fig.add_subplot(121)

	ax1.scatter(rv_inst.get_velocity_vec(), rv_inst.get_radius(), c='b', s=100)

	ax1.set_xlabel('Velocity (km/s)')
	ax1.set_ylabel('Radius (deg)')

	ax2 = fig.add_subplot(122)

	sc = ax2.scatter(rv_inst.get_velocity_vec(), rv_inst.get_radius(), 
					 c=rv_inst.get_normalized_memberships(), s=100)
	cbar = fig.colorbar(sc)
	cbar.set_label(r'$P_M$')

	ax2.set_xlabel('Velocity (km/s)')
	# ax2.set_ylabel('Radius (deg)')
	if show:
		plt.show()

def rvm_plot (rv_inst, fig_name='rad_met_vel', show=True):
	fig = plt.figure(fig_name)
	ax1 = fig.add_subplot(121)

	ax1.scatter(rv_inst.get_velocity_vec(), rv_inst.get_param2_vec(), c='b', s=100)

	ax1.set_xlabel('Velocity (km/s)')
	ax1.set_ylabel('Metallicity (dex)')

	ax2 = fig.add_subplot(122)

	sc = ax2.scatter(rv_inst.get_velocity_vec(), rv_inst.get_param2_vec(), 
					 c=rv_inst.get_normalized_memberships(), s=100)
	cbar = fig.colorbar(sc)
	cbar.set_label(r'$P_M$')

	ax2.set_xlabel('Velocity (km/s)')
	# ax2.set_ylabel('Radius (deg)')
	if show:
		plt.show()




###############################################################################
# Simulation 1: Radius and Velocity. Object slightly away from field

def simulation1 ():
	# object, simulates a globuar cluster
	object_velocities = np.random.normal(loc=-20,scale=8,size=200)
	object_vel_error = np.abs(np.random.normal(loc=0, scale=1, size=200))
	object_radius = np.abs(np.random.normal(loc=0, scale=0.3, size=200))


	# simulates the field of the Milky Way
	field_velocities = np.random.normal(loc=10, scale=50, size=200)
	field_vel_error = np.abs(np.random.normal(loc=0, scale=1, size=200))
	field_radius = np.abs(np.random.normal(loc=0, scale=0.3, size=200))

	# simulated data from telescope -- mix the object with the field
	velocity_data = np.concatenate((object_velocities, field_velocities))
	velocity_error_data = np.concatenate((object_vel_error, field_vel_error))
	radius_data = np.concatenate((object_radius, field_radius))


	# typically from an astronimical model (besancon model usually)
	field_model = np.random.normal(loc=0, scale=60, size=300)

	# use the algorithm
	rv = clumpy.EM(velocity_data, velocity_error_data, radius_data, 
					contam_model=field_model, 
					fit_model=None)


	# print the results of the model and what the algorithm found 
	# with the contamination.
	print('----------------------') 
	print('SIMULATION 1:')
	print('Mean Velocity')
	print("    model : {}".format(np.mean(object_velocities)))
	print("    recovered : {}".format(rv.get_mean_velocity()))

	print('Dispersion Velocity (sigma)')
	print("    model : {}".format(np.std(object_velocities)))
	print("    recovered : {}".format(rv.get_dispersion()))
	print('----------------------') 

	return rv

###############################################################################
# Simulation 2: Radius and Velocity and a Second Parameter (usually Metallicity). 
# Object slightly away from field

def simulation2 ():
	# object, simulates a globuar cluster
	object_velocities = np.random.normal(loc=-20,scale=8,size=200)
	object_vel_error = np.abs(np.random.normal(loc=0, scale=1, size=200))
	object_radius = np.abs(np.random.normal(loc=0, scale=0.3, size=200))
	object_metal = np.random.normal(loc=-2, scale=0.1, size=200)
	object_metal_error = np.abs(np.random.normal(loc=0, scale=0.1, size=200))


	# simulates the field of the Milky Way
	field_velocities = np.random.normal(loc=10, scale=50, size=200)
	field_vel_error = np.abs(np.random.normal(loc=0, scale=1, size=200))
	field_radius = np.abs(np.random.normal(loc=0, scale=0.3, size=200))
	field_metal = np.random.normal(loc=0, scale=2, size=200)
	field_metal_error = np.abs(np.random.normal(loc=0, scale=0.1, size=200))


	# simulated data from telescope -- mix the object with the field
	velocity_data = np.concatenate((object_velocities, field_velocities))
	velocity_error_data = np.concatenate((object_vel_error, field_vel_error))
	radius_data = np.concatenate((object_radius, field_radius))
	metal_data = np.concatenate((object_metal, field_metal))
	metal_data_error = np.concatenate((object_metal_error, field_metal_error))


	# typically from an astronimical model (besancon model usually)
	field_model = np.random.normal(loc=0, scale=60, size=300)

	# use the algorithm
	rvm = clumpy.EMFull(velocity_data, velocity_error_data, radius_data, 
					contam_model=field_model, param2_vec=metal_data, 
					param2_err_vec=metal_data_error, 
					fit_model=None)


	# print the results of the model and what the algorithm found 
	# with the contamination.
	print('----------------------') 

	print('SIMULATION 2:')
	print('Mean Velocity')
	print("    model : {}".format(np.mean(object_velocities)))
	print("    recovered : {}".format(rvm.get_mean_velocity()))

	print('Dispersion Velocity (sigma)')
	print("    model : {}".format(np.std(object_velocities)))
	print("    recovered : {}".format(rvm.get_dispersion()))

	print('Mean Metallicity')
	print("    model : {}".format(np.mean(object_metal)))
	print("    recovered : {}".format(rvm.get_mean_param2()))

	print('Dispersion Metallicity (sigma)')
	print("    model : {}".format(np.std(object_metal)))
	print("    recovered : {}".format(rvm.get_dispersion_param2()))

	print('----------------------') 

	return rvm





if __name__ == '__main__':


	import os,sys
	parentdir = os.path.dirname('../../')
	sys.path.insert(0,parentdir)
	
	import clumpy
	# Plot them
	# simulation 1
	rv = simulation1()
	rv_plot(rv, fig_name='simulation-1', show=False)


	# simulation 2
	# rvm = simulation2()
	# rv_plot(rvm, fig_name='simulation-2a', show=False)
	# rvm_plot(rvm, fig_name='simulation-2b', show=False)

	plt.show()










