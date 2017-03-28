import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import scipy.optimize as opt


def _rotation(x, y, velocity, step):
    line = np.array((-10, 10))

    # N = 0 (+DEC), E=90 (+RA)
    degree_steps = np.arange(180, 540 + step, step)
    degree_steps = degree_steps[::-1]

    degree_steps_for_plot = np.arange(0, 360 + step, step)

    rotation_vels = []
    for degree in degree_steps:
        line_theta = degree*(np.pi/180)
        new_line_x = line*np.cos(line_theta)
        new_line_y = line*np.sin(line_theta)

        position = (
            (new_line_x[0]-new_line_x[1]) *
            (y-new_line_y[1]) -
            (new_line_y[0]-new_line_y[1]) *
            (x-new_line_x[1]))

        mask_r = (position > 0)  # right side
        mask_l = (position < 0)  # left side

        velocity_difference = np.mean(
            velocity[mask_l]) - np.mean(velocity[mask_r])
        rotation_vels.append(velocity_difference)

        # sanity check, make sure it is rotating correctly
#         plt.plot(newLineX,newLineY,'k')
#         plt.scatter(x[maskR],y[maskR],s=100,c='c')
#         plt.scatter(x[maskL],y[maskL],s=100,c='m')
#         plt.xlabel('RA',fontsize=24)
#         plt.ylabel('DEC', fontsize=24)
#         plt.axis([-1,1,-1,1])
#         plt.show()
#     import ipdb;ipdb.set_trace()

    return np.array(rotation_vels), degree_steps_for_plot


def rotation(
        ra, dec, center_ra, center_dec, velocity, membership,
        step=10, plot=False):
    """
    Determine the rotation of your dataset. Only use the 'members' (p > 0.5) to
    derive the rotation.

    Parameters
    ----------
    ra : np.array
        right ascension vector
    dec : np.array
        declination vector
    center_ra : float
        center of the object ra
    center_dec : float
        center of the object dec
    velocity : np.array
        the velocity vector
    membership : np.array
        the membership vector
    step : float, int
        the step you wish to take in degrees
    plot : bool
        (default: False)

    Returns
    -------
    arot : float
        the maximum rotation
    arot_error : float
        the error on the maximum rotation
    pa : float (degrees)
        the position angle associated with the maximum rotation
    pa_error : fload (degrees)
        the error on the position angle from bootstraping
    v_rots : np.array
        the rotation at each position angle
    degs : np.array
        the degree values of each position angle
    result : tuple
        the result from the fit (arot,pa) for reconstructing the sine wave
    """
    member_mask = (membership > 0.50)

    line = np.array((-10, 10))

    x_points = (center_ra - ra)[member_mask]
    y_points = (center_dec - dec)[member_mask]
    v_points = velocity[member_mask]

    v_rots, degs = _rotation(
        x_points,
        y_points,
        v_points,
        step)

    # parameter guesses
    guess_arot = 3*np.std(v_rots)/(2**0.5)
    guess_pa = 0.
    guess = [guess_arot, guess_pa]

    # calculate the errors
    arot_error_array = []
    pa_error_array = []
    for i in xrange(100):
        idx = np.random.randint(
            0,
            len(x_points),
            size=len(x_points))

        v_rots_error, degs_error = _rotation(
            x_points[idx],
            y_points[idx],
            v_points[idx],
            step)
        #check the errors, with plots
        result = sine_fit_bellazzini(
            degs_error,
            v_rots_error,
            guess)

        arot_error_array.append(result[0])
        pa_error_array.append(result[1])

    # where the fit finds the maximum value each time
    arot_error = np.std(np.abs(arot_error_array))
    pa_error = np.std(pa_error_array)

    result = sine_fit_bellazzini(degs, v_rots, guess)
    arot, pa = result

    if plot:
        fig = plt.figure('rotation')
        ax = fig.add_subplot(111)
        ax.scatter(degs, v_rots, c='k', s=100)
        ax.plot(degs, sine_model(result, degs), 'b')
        ax.plot(degs, sine_model(guess, degs), 'r')  # guess
        ax.xlabel('Position Angle (deg)')
        ax.ylabel('Diff. in mean V_r (km/s)')
        ax.axis([-2, 362, plt.ylim()[0], plt.ylim()[1]])
        plt.show()

    return (abs(arot), arot_error, pa*(180./np.pi),
            pa_error*(180/np.pi), v_rots, degs, result)


# ########################################################################### #
# two parameter method
def sine_model(params, xpts):
    A, phi = params
    return A*np.sin(xpts*(np.pi/180.) + phi)


def error_function(params, xpts, ypts):
    yfit = sine_model(params, xpts)
    chi2 = np.sqrt(np.sum((ypts-yfit)**2))
    deg_freedom = (len(ypts)-len(params))
    return chi2/deg_freedom


def sine_fit_bellazzini(xpts, ypts, params0):
    kws = dict(args=(xpts, ypts),
               full_output=0,
               disp=0)
    result = opt.fmin(error_function, params0, **kws)
    return result
