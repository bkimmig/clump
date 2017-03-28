import numpy as np

from . import functions_em as f
from .py3 import *


def em_vr(velocity, velocity_error, contam_model,
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
    #   fit_model = np.ones(len(velocity))

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

        mean_velocity = f.mean(
            np.sqrt(variance_velocity),
            p_m,
            velocity,
            velocity_error,
            fit_model)

        # append the first iteration (guess)
        if i == 0:
            mean_velocity_iter.append(mean_velocity)

        variance_velocity = f.variance(
            mean_velocity,
            np.sqrt(variance_velocity),
            p_m,
            velocity,
            velocity_error,
            fit_model)

        # raw membership probabilities
        p_mem = f.p_normalized(
            np.sqrt(variance_velocity),
            mean_velocity,
            velocity,
            velocity_error,
            fit_model)

        # normalized membership probabilities
        p_m = f.normalized_probs(p_mem, p_non, p_a)

        #no nans in prob, make sure
        no_nans = np.isnan(p_m)

        p_m = np.where(no_nans == True, 0, p_m)

        # calculae the a priori
        p_a = f.pav(p_m)

        #calculate the likelihood
        log_like = f.log_likelihood(p_m, p_a, p_mem, p_non)

        #make the iteration vectors
        mean_velocity_iter.append(mean_velocity)
        variance_velocity_iter.append(variance_velocity)
        loglike_iter.append(log_like)
        i += 1

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


def em_ivr(velocity, velocity_error, param2, param2_error,
           contam_model, fit_model=None, iterations=50, full_output=False):
    """
    The em method that fits for two paramters and encorporates
    the spatial distribution.

    Parameters
    ----------
    velocity : np.array
    velocity_error : np.array
    param2 : np.array
    param2_error : np.array
    contam_model : np.array
    fit_model : np.array

    Returns
    -------
    mean_velocity : float
        Average velocity determined.
    velocity_dispersion : float
        Velocity dispersion determined (np.sqrt(variance_velocity))
    mean_param2 : float
        Average of param2
    param2_dispersion : float
        The dispersion in param 2
    mean_param2_non : float
        Average of the non param2 (field)
    param2_non_dispersion : float
        The dispersion in non param 2 (field)
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
    param2_iteration : np.array
        Array showing the mean param2 at each iteration
    param2_sig_iteration,
        Array showing the disperison param2 at each iteration
    param2_non_iteration,
        Array showing the field param2 mean at each iteration
    param2_sig_non_iteration : np.array
        Array showing the field param2 dispersion at each iteration


    """
    #initial guesses
    p_m = np.ones(len(velocity))*0.5
    p_a = np.ones(len(velocity))*0.5
    variance_velocity = 2.0**2
    variance_param2 = 2.0**2
    variance_param2_non = 2.0**2

    # contamination probability vector
    p_non_vel = f.p_contamination_non(velocity, contam_model)  # p_bes

    # iteration storage
    mean_velocity_iter = []
    variance_velocity_iter = [variance_velocity]
    loglike_iter = [0]

    param2_iteration = []
    param2_sig_iteration = [variance_param2]

    param2_non_iteration = []
    param2_sig_non_iteration = [variance_param2_non]

    # start iterations
    i = 0
    while i < iterations:

        mean_velocity = f.mean(
            np.sqrt(variance_velocity),
            p_m,
            velocity,
            velocity_error,
            fit_model)

        variance_velocity = f.variance(
            mean_velocity,
            np.sqrt(variance_velocity),
            p_m,
            velocity,
            velocity_error,
            fit_model)

        # the member fitted variance
        mean_param2 = f.mean(
            np.sqrt(variance_param2),
            p_m,
            param2,
            param2_error)  # no model

        # the member fitted variance
        variance_param2 = f.variance(
            mean_param2,
            np.sqrt(variance_param2),
            p_m,
            param2,
            param2_error)  # no model

        # the non member param2 fitted mean
        mean_param2_non = f.mean_non(
            np.sqrt(variance_param2),
            p_m,
            param2,
            param2_error)

        # the non member param2 fitted variance
        variance_param2_non = f.variance_non(
            mean_param2,
            np.sqrt(variance_param2),
            p_m,
            param2,
            param2_error)  # no model

        # append the first iteration (guess)
        if i == 0:
            mean_velocity_iter.append(mean_velocity)
            param2_iteration.append(mean_param2)
            param2_non_iteration.append(mean_param2_non)

        # raw membership probabilities
        p_mem_vel = f.p_normalized(
            np.sqrt(variance_velocity),
            mean_velocity,
            velocity,
            velocity_error,
            fit_model)

        # multiply your membership probs from each
        # member distribution
        p_mem = f.p_normalized(
            np.sqrt(variance_param2),
            mean_param2,
            param2,
            param2_error)*p_mem_vel  # no fit model

        # multiply your non-membership probs from each
        # non-member distribution
        p_non = f.p_normalized(
            np.sqrt(variance_param2_non),
            mean_param2_non,
            param2,
            param2_error)*p_non_vel  # no fit model

        # normalized membership probabilities
        p_m = f.normalized_probs(p_mem, p_non, p_a)

        #no nans in prob, make sure
        no_nans = np.isnan(p_m)

        p_m = np.where(no_nans == True, 0, p_m)

        # calculae the a priori
        p_a = f.pav(p_m)

        #calculate the likelihood
        log_like = f.log_likelihood(p_m, p_a, p_mem, p_non)

        #make the iteration vectors
        mean_velocity_iter.append(mean_velocity)
        variance_velocity_iter.append(variance_velocity)
        loglike_iter.append(log_like)
        param2_iteration.append(mean_param2)
        param2_sig_iteration.append(variance_param2)
        param2_non_iteration.append(mean_param2_non)
        param2_sig_non_iteration.append(variance_param2_non)

        i += 1

    if full_output:
        return_ = [mean_velocity,
                   np.sqrt(variance_velocity),
                   mean_param2,
                   np.sqrt(variance_param2),
                   mean_param2_non,
                   np.sqrt(variance_param2_non),
                   log_like,
                   p_m,
                   p_mem,
                   p_non,
                   p_a,
                   mean_velocity_iter,
                   variance_velocity_iter,
                   loglike_iter,
                   param2_iteration,
                   param2_sig_iteration,
                   param2_non_iteration,
                   param2_sig_non_iteration]
    else:
        return_ = [mean_velocity,
                   np.sqrt(variance_velocity),
                   mean_param2,
                   np.sqrt(variance_param2),
                   mean_param2_non,
                   np.sqrt(variance_param2_non),
                   log_like]

    return return_


class EM(object):
    """
    The expecatation-maximization algorithm. You should not have velocities
    that in your sample that lie outside the contamination model range.

    From Walker 2009

    Will do two parameter fits if vectors are included
    """
    def __init__(self, velocity, velocity_error, radius, contam_model,
                 fit_model=None):

        """
        Parameters
        ----------

        velocity: np.array
            The velocities of the objects in your sample
        velocity_error: np.array
            The errors on the velocities of the objects in your sample
        radius: np.array
            The distace from the center of the object for each
            associated velocity in your sample
        contam_model: np.array
            Velocity distribution describing the contamination of the
            foreground and background. Typically a
            Besancon Model (Robin et al. 2003)
        fit_model: np.array
            The model you wish to use to fit you data for a dispersion.
            Typically a King dispersion profile. These points
            should be in line with your data.

        """
        # initalize everything
        self.velocity = velocity
        self.velocity_error = velocity_error
        self.radius = radius
        self.contam_model = contam_model
        self.fit_model = fit_model

        # sort the data radially
        self.sort_args = np.argsort(self.radius)

        # all the parameters
        self.mean_velocity = None
        self.mean_velocity_error = None

        self.dispersion = None
        self.dispersion_error = None

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
        self.em()

    def get_sort_args(self):
        return self.sort_args

    def get_velocity_vec(self, _sorted=True):
        if _sorted:
            return self.velocity[self.sort_args]
        else:
            return self.velocity

    def get_velocity_error_vec(self, _sorted=True):
        if _sorted:
            return self.velocity_error[self.sort_args]
        else:
            return self.velocity_error

    def get_radius(self, _sorted=True):
        if _sorted:
            return self.radius[self.sort_args]
        else:
            return self.radius

    def get_contam_model(self):
        return self.contam_model

    def get_fit_model(self, _sorted=True):
        if self.fit_model is not None:
            if _sorted:
                return self.fit_model[self.sort_args]
        else:
            return self.fit_model

    def get_mean_velocity(self):
        return self.mean_velocity

    def get_mean_velocity_error(self):
        return self.mean_velocity_error

    def get_dispersion(self):
        return self.dispersion

    def get_dispersion_error(self):
        return self.dispersion_error

    def get_normalized_memberships(self):
        return self.normalized_memberships

    def get_p_mem(self):
        return self.p_mem

    def get_p_non(self):
        return self.p_non

    def get_p_a(self):
        return self.p_a

    def get_iteration_mean_velocity(self):
        return self.mean_velocity_iter

    def get_iteration_variance_velocity(self):
        return self.variance_velocity_iter

    def get_iteration_loglike(self):
        return self.loglike_iter

    def em(self):
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

    def bootstrap_errors(self, iterations=100):
        """
        Derive the errors on the derived parameters of your data.
        Method is bootstrap resampling.

        """
        dispersions = []
        velocities = []

        length = len(self.get_velocity_vec())
        i = 0
        while i < iterations:  # 100 for paper
            error_idx = np.random.randint(0, length, length)
            vel_err = self.get_velocity_error_vec()[error_idx]
            vel = self.get_velocity_vec()[error_idx]
            rad = self.get_radius()[error_idx]
            args_ = np.argsort(rad)
            output = em_vr(vel[args_],
                           vel_err[args_],
                           self.get_contam_model(),
                           fit_model=self.get_fit_model(),
                           iterations=50,
                           full_output=True)
            # import ipdb;ipdb.set_trace()
            if np.isnan(output[0]) is True or np.isnan(output[1]) is True:
                continue
            velocities.append(output[0])
            dispersions.append(output[1])
            i += 1

        self.bootstrap_dispersions = np.array(dispersions)
        self.bootstrap_velocities = np.array(velocities)
        self.mean_velocity_error = np.std(velocities)
        self.dispersion_error = np.std(dispersions)

        return 1


class EMFull (object):
    """
    The expecatation-maximization algorithm. You should not have velocities
    that in your sample that lie outside the contamination model range.

    From Walker 2009

    Will do two parameter fits if vectors are included
    """
    def __init__(self, velocity, velocity_error, radius, contam_model,
                 fit_model=None, param2_vec=None, param2_err_vec=None):

        """
        Parameters
        ----------

        velocity: np.array
            The velocities of the objects in your sample
        velocity_error: np.array
            The errors on the velocities of the objects in your sample
        radius: np.array
            The distace from the center of the object for each associated
            velocity in your sample
        contam_model: np.array
            Velocity distribution describing the contamination of the
            foreground and background.
            Typically a Besancon Model (Robin et al. 2003)
        fit_model: np.array
            The model you wish to use to fit you data for a dispersion.
            Typically a King dispersion profile. These points should
            be in line with your data
        param2_vec: np.array
            A second parameter you wish to fit. e.g. Equivalent Width.
        param2_err_vec: np.array
            The errors on the second parameter you wish to fit. e.g.
            Equivalent Width errors.

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

        self.mean_param2_non = None
        self.mean_param2_non_error = None

        self.dispersion_param2_non = None
        self.dispersion_param2_non_error = None

        self.loglike = None
        self.normalized_memberships = None
        self.p_mem = None
        self.p_non = None
        self.p_a = None

        # add more here for param 2
        self.mean_velocity_iter = None
        self.variance_velocity_iter = None
        self.loglike_iter = None
        self.mean_param2_iter = None
        self.variance_param2_iter = None
        self.mean_param2_iter_non = None
        self.variance_param2_iter_non = None

        # run the algorithm
        self.em()

    def get_sort_args(self):
        return self.sort_args

    def get_velocity_vec(self, _sorted=True):
        if _sorted:
            return self.velocity[self.sort_args]
        else:
            return self.velocity

    def get_velocity_error_vec(self, _sorted=True):
        if _sorted:
            return self.velocity_error[self.sort_args]
        else:
            return self.velocity_error

    def get_radius(self, _sorted=True):
        if _sorted:
            return self.radius[self.sort_args]
        else:
            return self.radius

    def get_contam_model(self):
        return self.contam_model

    def get_fit_model(self, _sorted=True):
        if self.fit_model is not None:
            if _sorted:
                return self.fit_model[self.sort_args]
        else:
            return self.fit_model

    def get_param2_vec(self, _sorted=True):
        if self.param2_vec is not None:
            if _sorted:
                return self.param2_vec[self.sort_args]
        else:
            return self.param2_vec

    def get_param2_error_vec(self, _sorted=True):
        if self.param2_err_vec is not None:
            if _sorted:
                return self.param2_err_vec[self.sort_args]
        else:
            return self.param2_err_vec

    def get_mean_velocity(self):
        return self.mean_velocity

    def get_mean_velocity_error(self):
        return self.mean_velocity_error

    def get_dispersion(self):
        return self.dispersion

    def get_dispersion_error(self):
        return self.dispersion_error

    def get_mean_param2(self):
        return self.mean_param2

    def get_mean_param2_error(self):
        return self.mean_param2_error

    def get_dispersion_param2(self):
        return self.dispersion_param2

    def get_dispersion_param2_error(self):
        return self.dispersion_param2_error

    def get_normalized_memberships(self):
        return self.normalized_memberships

    def get_p_mem(self):
        return self.p_mem

    def get_p_non(self):
        return self.p_non

    def get_p_a(self):
        return self.p_a

    def get_iteration_mean_velocity(self):
        return self.mean_velocity_iter

    def get_iteration_variance_velocity(self):
        return self.variance_velocity_iter

    def get_iteration_loglike(self):
        return self.loglike_iter

    def em(self):
        output = em_ivr(
            self.get_velocity_vec(),
            self.get_velocity_error_vec(),
            self.get_param2_vec(),
            self.get_param2_error_vec(),
            self.get_contam_model(),
            fit_model=self.get_fit_model(),
            iterations=50,
            full_output=True)

        # values, derived
        self.mean_velocity = output[0]
        self.dispersion = output[1]

        self.mean_param2 = output[2]
        self.dispersion_param2 = output[3]

        self.mean_param2_non = output[4]
        self.dispersion_param2_non = output[5]

        self.loglike = output[6]

        # final vectors
        self.normalized_memberships = output[7]
        self.p_mem = output[8]
        self.p_non = output[9]
        self.p_a = output[10]

        # iterations
        self.mean_velocity_iter = output[11]
        self.variance_velocity_iter = output[12]
        self.loglike_iter = output[13]
        self.mean_param2_iter = output[14]
        self.variance_param2_iter = output[15]
        self.mean_param2_iter_non = output[16]
        self.variance_param2_iter_non = output[17]

    def bootstrap_errors(self, iterations=100):
        """
        Derive the errors on the derived parameters of your data.
        Method is bootstrap resampling.

        """
        dispersions = []
        velocities = []

        mean_param2 = []
        disp_param2 = []

        mean_param2_non = []
        disp_param2_non = []

        length = len(self.get_velocity_vec())
        i = 0
        while i < iterations:  # 100 for paper
            error_idx = np.random.randint(0, length, length)
            vel_err = self.get_velocity_error_vec()[error_idx]
            vel = self.get_velocity_vec()[error_idx]
            p2 = self.get_param2_vec()[error_idx]
            p2_err = self.get_param2_error_vec()[error_idx]
            rad = self.get_radius()[error_idx]
            args_ = np.argsort(rad)

            output = em_ivr(
                vel[args_],
                vel_err[args_],
                p2[args_],
                p2_err[args_],
                self.get_contam_model(),
                fit_model=self.get_fit_model(),
                iterations=50,
                full_output=True)

            if np.isnan(output[0]) is True or np.isnan(output[1]) is True:
                continue
            velocities.append(output[0])
            dispersions.append(output[1])
            mean_param2.append(output[2])
            disp_param2.append(output[3])
            mean_param2_non.append(output[4])
            disp_param2_non.append(output[5])
            
            i += 1

        self.bootstrap_dispersions = np.array(dispersions)
        self.bootstrap_velocities = np.array(velocities)
        self.bootstrap_mean_param2 = np.array(mean_param2)
        self.bootstrap_disp_param2 = np.array(disp_param2)
        self.bootstrap_mean_param2_non = np.array(mean_param2_non)
        self.bootstrap_disp_param2_non = np.array(disp_param2_non)

        self.mean_velocity_error = np.std(velocities)
        self.dispersion_error = np.std(dispersions)
        self.mean_param2_error = np.std(mean_param2)
        self.dispersion_param2_error = np.std(disp_param2)
        self.mean_param2_error_non = np.std(mean_param2_non)
        self.dispersion_param2_error_non = np.std(disp_param2_non)

        return 1
