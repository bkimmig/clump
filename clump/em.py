import numpy as np

from . import functions_em as f 
 

def em_vr (velocity, velocity_error, contam_model, 
			fit_model=None, iterations=50, full_output=False):
	"""
	The application of the Expectation Maximization method. 

	Parameters
	----------
	velocity : np.array
		Array of velocities sorted by distance from the center of 
		the object.
	velocity_error : np.array
		Array of velocity errors corresponding to the velocity array
	contamination_model : np.array
		Array of velocities describing the contamination in the field, 
		typically a Besancon model (Robin et al. (2003))
	fit_model : np.array, optional
		A profile that corresponds to the the velocities. Originally 
		a King dispersion profile. Must be same length as velocity 
		array and correspond to the normalized model dispersion at 
		the particular radius of the star.
		Default is None
	iterations : int, optional
		The number of iterations you wish to run the algorithm for.
		It converges quickly, typically faster than 50 iterations, 
		but due to the speed of the alg. we set the default to 50.
	full_output : bool, optional
		If True, will return info on iterations and membership 
		vectors (p_normalized, p_mem, p_non, p_a, mean_velocity_iter,
		variance_velocity_iter, log_like_iter). default is False

	Returns
	-------
	mean_velocity : float
		Average velocity determined.
	velocity_dispersion : float
		Velocity dispersion determined (np.sqrt(variance_velocity))
	log_like : float
		Maximum log likelihood corresponding to the parameters
		derived.
	p_m : np.array
		Array of normalized membership probabilities. (final)
	p_mem : np.array
		Array of raw membership probabilities. (final)
	p_non : np.array
		Array of raw non membership probabilities. (final)
	p_a : np.array
		Array from monotonic non increasing function. (final)
	mean_velocity_iter : np.array
		Array showing the velocity at each iteration.
	variance_velocity_iter : np.array
		Array showing the variance at each iteration.
	log_like_iter : np.array 
		Array showing the log_likelihood at each iteration

	"""

	# fitting model is not necessary
	# if fit_model == None:
	# 	fit_model = np.ones(len(velocity))

	#initial guesses
	p_m = np.ones(len(velocity))*0.5
	p_a = np.ones(len(velocity))*0.5
	variance_velocity = 2.0**2

	# contamination probability vector
	p_non = f.p_contamination_non(velocity, contam_model)             
	        
	# iteration storage
	mean_velocity_iter = []
	variance_velocity_iter = [variance_velocity]
	loglike_iter = [0]

	# start iterations
	i = 0
	while i < iterations:

		mean_velocity = f.mean(np.sqrt(variance_velocity),
								p_m,
								velocity,
								velocity_error,
								fit_model)

	    # append the first iteration (guess)
		if i == 0:
			mean_velocity_iter.append(mean_velocity)


		variance_velocity = f.variance(mean_velocity, 
		                             	np.sqrt(variance_velocity), 
		                             	p_m, 
		                             	velocity, 
		                             	velocity_error, 
		                             	fit_model)


		# raw membership probabilities
		p_mem = f.p_normalized(np.sqrt(variance_velocity), 
								   mean_velocity, 
								   velocity, 
								   velocity_error, 
								   fit_model)


		# normalized membership probabilities  
		p_m = f.normalized_probs(p_mem, p_non, p_a)         


		#no nans in prob, make sure
		no_nans = np.isnan(p_m)

		p_m = np.where(no_nans==True, 0, p_m) 

		# calculae the a priori
		pa = f.pav(p_m)

		#calculate the likelihood
		log_like = f.log_likelihood(p_m, p_a, p_mem, p_non)


		#make the iteration vectors
		mean_velocity_iter.append(mean_velocity)
		variance_velocity_iter.append(variance_velocity)
		loglike_iter.append(log_like)
		i+=1

	if full_output: 
		return_ = [mean_velocity,
				   np.sqrt(variance_velocity),
				   log_like,
				   p_m,
				   p_mem,
				   p_non,
				   p_a,
				   mean_velocity_iter,
				   variance_velocity_iter,
				   loglike_iter]
	else:
		return_ = [mean_velocity,
				   np.sqrt(variance_velocity),
				   log_like]

	return return_	



def em_vr_ew (velocity, velocity_error, radius, param2, param2_error, 
			contam_model, fit_model):
	pass


class EM (object):
	"""
	The expecatation-maximization algorithm.

	From Walker 2009

	Will do two parameter fits if vectors are included
	"""


	def __init__ (self, velocity, velocity_error, radius, contam_model, 
					fit_model=None, param2_vec=None, param2_err_vec=None):

		"""
		Parameters
		----------

		velocity: np.array
			The velocities of the objects in your sample
		velocity_error: np.array
			The errors on the velocities of the objects in your sample
		radius: np.array
			The distace from the center of the object for each associated velocity
			in your sample
		contam_model: np.array
			Velocity distribution describing the contamination of the foreground and
			background. Typically a Besancon Model (Robin et al. 2003)
		fit_model: np.array
			The model you wish to use to fit you data for a dispersion. Typically a 
			King dispersion profile. These points should be in line with your data
		param2_vec: np.array
			A second parameter you wish to fit. e.g. Equivalent Width.
		param2_err_vec: np.array
			The errors on the second parameter you wish to fit. e.g. Equivalent Width
			errors.

		"""
		
		# initalize everything
		self.velocity = velocity
		self.velocity_error = velocity_error
		self.radius = radius
		self.contam_model = contam_model
		self.fit_model = fit_model
		
		self.param2_vec = param2_vec
		self.param2_err_vec = param2_err_vec

		# sort the data radially
		self.sort_args = np.argsort(self.radius)

		# all the parameters 
		self.mean_velocity = None
		self.mean_velocity_error = None
		
		self.dispersion = None
		self.dispersion_error = None

		self.mean_param2 = None
		self.mean_param2_error = None

		self.dispersion_param2 = None
		self.dispersion_param2_error = None

		self.loglike = None
		self.normalized_memberships = None
		self.p_mem = None
		self.p_non = None
		self.p_a = None
		
		# add more here for param 2
		self.mean_velocity_iter = None
		self.variance_velocity_iter = None
		self.loglike_iter = None


		# run the algorithm
		if self.param2_vec == None:
			self.em()
		else:
			self.em_2param()

	def get_sort_args (self):
		return self.sort_args

	def get_velocity_vec (self, _sorted=True):
		if _sorted:
			return self.velocity[self.sort_args]
		else:
			return self.velocity

	def get_velocity_error_vec (self, _sorted=True):
		if _sorted:
			return self.velocity_error[self.sort_args]
		else:
			return self.velocity_error

	def get_radius (self, _sorted=True):
		if _sorted:
			return self.radius[self.sort_args]
		else:
			return self.radius

	def get_contam_model (self):
		return self.contam_model


	def get_fit_model(self, _sorted=True):
		if self.fit_model != None:
			if _sorted:
				return self.fit_model[self.sort_args]
		else:
			return self.fit_model

	def get_param2_vec(self, _sorted=True):
		if self.param2_vec != None:
			if _sorted:
				return self.param2_vec[self.sort_args]
		else:
			return self.param2_vec

	def get_param2_error_vec(self, _sorted=True):
		if self.param2_err_vec != None:
			if _sorted:
				return self.param2_err_vec[self.sort_args]
		else:
			return self.param2_err_vec

	def get_mean_velocity (self):
		return self.mean_velocity

	def get_mean_velocity_error (self):
		return self.mean_velocity_error

	def get_dispersion (self):
		return self.dispersion

	def get_dispersion_error (self):
		return self.dispersion_error

	def get_mean_param2 (self):
		return self.mean_param2
	
	def get_mean_param2_error (self):
		return self.mean_param2_error

	def get_dispersion_param2 (self):
		return self.mean_param2
	
	def get_dispersion_param2_error (self):
		return self.mean_param2_error

	def get_normalized_memberships (self):
		return self.normalized_memberships

	def get_p_mem (self):
		return self.p_mem

	def get_p_non (self):
		return self.p_non

	def get_p_a (self):
		return self.p_a

	def get_iteration_mean_velocity (self):
		return self.mean_velocity_iter

	def get_iteration_variance_velocity (self):
		return self.variance_velocity_iter

	def get_iteration_loglike (self):
		return self.loglike_iter

	def em (self):
		output = em_vr(self.get_velocity_vec(), 
					   self.get_velocity_error_vec(), 
					   self.get_contam_model(), 
					   fit_model=self.get_fit_model(), 
					   iterations=50, 
					   full_output=True)

		# values, derived
		self.mean_velocity = output[0]
		self.dispersion = output[1]
		self.loglike = output[2]
		
		# final vectors
		self.normalized_memberships = output[3]
		self.p_mem = output[4]
		self.p_non = output[5]
		self.p_a = output[6]
		
		# iterations
		self.mean_velocity_iter = output[7]
		self.variance_velocity_iter = output[8]
		self.loglike_iter = output[9]			



	def em_2param (self):
		pass







































