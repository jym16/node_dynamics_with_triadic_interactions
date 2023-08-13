"""
Computation module.

This module provides functions for computations.
"""

# Import packages
import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import iqr

def create_node_edge_incidence_matrix(edge_list):
    """Create a node-edge incidence matrix B from a given edge list.

    Parameters
    ----------
    edge_list : list of tuples  (i, j)  * i < j
        The list of edges (i, j).

    Returns
    -------
    B : numpy.ndarray of shape (n_nodes, n_edges)
        The node-edge incidence matrix.
    
    """
    num_edges = len(edge_list) # the number of edges
    if num_edges == 0: # if the number of edges is zero, return None
        return None
    b_ij = [-1] * num_edges + [1] * num_edges # the (i,j)-th element of B in a sequence
    row_i = [e[0]-1 for e in edge_list] + [e[1]-1 for e in edge_list] # the row indices (i) of the (i, j)-th element
    col_j = [l for l in range(num_edges)] * 2 # the column indices (i) of the (i, j)-th element
    B = csc_matrix((np.array(b_ij), (np.array(row_i), np.array(col_j))), dtype=np.int8)
    return B.toarray()

def extract_by_std(X, std=3.0):
    """Extract the data within a given number of standard deviations from its mean.
    
    Parameters
    ----------
    X : numpy.ndarray of shape (n_observations,)
        The data.
    std : float, optional
        (default = 3.0)
        The number of standard deviations to extract.
    
    Returns
    -------
    X_min : float
        The minimum value of the core range.
    X_max : float
        The maximum value of the core range.
    """
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_min = X_mean - std * X_std
    X_max = X_mean + std * X_std
    return X_min, X_max

def freedman_diaconis_rule(data, power=1. / 3., factor=2., trim=1):
    """Compute the number of bins using the Freedman-Diaconis rule.
    
    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations,)
        The data.
    power : float, optional
        (default = 1. / 3.)
        The power of the number of observations in the denominator.
    factor : float, optional
        (default = 2.)
        The factor to multiply the width of bins.
    trim : int, optional
        (default = 1)
        The ratio of the number of observations to trim from each end of the data.
    
    Returns
    -------
    bins_edges : numpy.ndarray of shape (n_bins,)
        The bins edges.
    
    """
    if data.ndim == 0 or data.ndim > 2:
        raise ValueError('The data must be a 1D or 2D array.')
    elif data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
        # Get the number of observations
        n_observations = data.shape[0]

        if trim > 1:
            n_observations = int(n_observations // trim)
        
        # Compute the interquartile range
        IQR = iqr(data, rng=(25, 75))
        
        # Compute the width of bins according to the Freedman-Diaconis rule
        width = (factor * IQR) / np.power(n_observations, power)

        # Generate the bins
        x_abs_max = np.max(np.abs(data))
        idx_upper = int(x_abs_max / width + 1)
        bin_edges = np.linspace(-idx_upper * width, idx_upper * width, 2 * idx_upper + 1)

        return bin_edges
    else:
        # Get the number of observations and variables
        n_observations, n_variables = data.shape

        # Initialise the bins edges
        bin_edges = []

        for i in range(n_variables):
            # Compute the interquartile range
            IQR = iqr(data[:, i], rng=(25, 75)) 
            
            # Compute the width of bins according to the Freedman-Diaconis rule
            width = (factor * IQR) / np.power(n_observations, power)

            bin_edges.append(np.min(data[:, i]), np.max(data[:, i]))
            assert(width > 0.)

            # Generate the bins
            x_abs_max = np.max(np.abs(data[:, i]))
            idx_upper = int(x_abs_max / width + 1)
            _bin_edges = np.linspace(-idx_upper * width, idx_upper * width, 2 * idx_upper + 1)

            bin_edges.append(_bin_edges)
        
        return bin_edges

def discretise(X, n_bins='fd'):
    """Discretise the time series data.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_observation, n_variables)
        The data matrix.
    n_bins : int or str, optional
        (default = 'fd')
        The number of bins or the method to compute the number of bins.
        - 'fd' : Freedman-Diaconis rule
    
    Returns
    -------
    X_discrete : numpy.ndarray of shape (n_observation, n_variables)
        The discretised data matrix.
    bins : list of numpy.ndarray of shape (n_bins,)
        The list of the bins of the values.
    
    """
    # Get the number of nodes
    n_variables = X.shape[1]

    # Compute the number of bins
    if isinstance(n_bins, str) and n_bins == 'fd':
        bin_edges = freedman_diaconis_rule(X)
    elif isinstance(n_bins, int):
        bin_edges = [np.linspace(-np.max(np.abs(X[v])), np.max(np.abs(X[v])), n_bins+1) for v in range(n_variables)]
    else:
        raise ValueError('The argument n_bins must be an integer or "fd".')
    
    # Get the alphabet of each node
    X_discrete = np.zeros(X.shape, dtype=np.int8) - 1

    # For each node
    for v in range(n_variables):
        # Convert the time series data to the alphabet
        X_discrete[:, v] = np.digitize(X[:, v], bin_edges[v]) - 1
    
    return X_discrete, bin_edges

def estimate_pdf(data, bins='fd', method='hist'):
    """Estimate the probability density function of the data by the histogram method.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations,) or (n_observations, 1)
        The data.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n : The number of bins for the variable.
    method : str, optional
        (default = 'hist')
        The method to estimate the probability density function (pdf).
        - 'hist' : The pdf is estimated by the histogram method.
        - 'kde' : The pdf is estimated by the kernel density estimation method.
    
    Returns
    -------
    P : numpy.ndarray of shape (n_bins, n_variables)
        The estimated probability density function of the data.
    X : numpy.ndarray of shape (n_bins, n_variables)
        The bin centers for variables.
    
    """
    if data.ndim > 2:
        raise ValueError(
                'The data must be a 1D array or 2D array. '
            )
    elif data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
        if method == 'hist':        
            if isinstance(bins, str) and bins == 'fd':
                _bins = freedman_diaconis_rule(data.flatten())
            elif isinstance(bins, np.ndarray):
                _bins = bins
            elif isinstance(bins, int):
                max_amp = np.max(np.abs(data))
                _bins = np.linspace(-max_amp, max_amp, bins+1)
                _bins = bins
            else:
                raise ValueError('Invalid bins.')

            # Create histogram
            pdf, bin_edges = np.histogram(
                data, 
                bins=_bins,
                density=True
            )
            
            # Calculate the x values corresponding to each bin
            x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            return pdf, x
        
        else:
            raise ValueError('The method not implemented.')
    elif data.ndim == 2:
        if method == 'hist':
            # Get the number of samples
            n_samples = data.shape[1]

            if isinstance(bins, str) and bins == 'fd':
                _bins = freedman_diaconis_rule(
                    data.flatten(), 
                    trim=n_samples
                )
                n_bins = len(_bins) - 1
            elif isinstance(bins, np.ndarray):
                _bins = bins
                n_bins = len(_bins) - 1
            elif isinstance(bins, int):
                max_amp = np.max(np.abs(data))
                _bins = np.linspace(-max_amp, max_amp, bins+1)
                n_bins = bins
            else:
                raise ValueError('Invalid bins.')


            PDF = np.zeros((n_bins, n_samples))
            
            for i in range(n_samples):
                # Create histogram
                pdf, bin_edges = np.histogram(
                    data[:, i], 
                    bins=_bins,
                    density=True
                )
                PDF[:, i] = pdf
            
            # Calculate the x values corresponding to each bin
            x = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            return PDF, x
        
        else:
            raise ValueError('The method not implemented.')

def estimate_pdf_joint(data, bins='fd', method='hist'):
    """Estimate the joint probability density function of the data by the histogram method.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations, n_variables)
        The data.
    bins : str or a sequence of int or int or list, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n_1, n_2, ..., n_n : The number of bins for each variable.
        - n : The number of bins for all variables.
        - list : The bin edges for each variable.
    method : str, optional
        (default = 'hist')
        The method to estimate the probability density function (pdf).
        - 'hist' : The pdf is estimated by the histogram method.
        - 'kde' : The pdf is estimated by the kernel density estimation method.
    
    Returns
    -------
    pdf_joint : numpy.ndarray of shape (n_bins, n_variables)
        The estimated probability density function of the data.
    x : list of numpy.ndarray of shape (n_bins-1,)
        The x values of the corresponding bins.
    
    """
    if method == 'hist':
        # Get the number of variables
        if data.ndim == 1:
            n_variables = 1
        else:
            n_variables = data.shape[1]

        if isinstance(bins, str) and bins == 'fd':
            _bins = [freedman_diaconis_rule(data[:, d]) for d in range(n_variables)]
        elif isinstance(bins, list):
            _bins = []
            if len(bins) != n_variables:
                raise ValueError('The length of bins must be equal to the number of variables.')
            for idx, item in enumerate(bins):
                if isinstance(item, np.ndarray):
                    _bins.append(item)
                elif isinstance(item, str) and item == 'fd':
                    _bins.append(freedman_diaconis_rule(data[:, idx]))
                elif isinstance(item, int):
                    _bins.append(np.linspace(np.min(data[:, idx]), np.max(data[:, idx]), num=item))
                else:
                    raise ValueError("Invalid bin type at index", idx, ".")
        elif isinstance(bins, int):
            _bins = [np.linspace(-np.max(np.abs(data[:, d])), np.max(np.abs(data[:, d])), num=bins+1) for d in range(n_variables)]
        else:
            raise ValueError('Invalid bins.')
        
        # Create histogram
        pdf, bin_edges = np.histogramdd(
            data,
            bins=_bins,
            density=True
        )
        
        # Calculate the x values corresponding to each bin
        for d in range(n_variables):
            bin_edges[d] = 0.5 * (bin_edges[d][1:] + bin_edges[d][:-1])
        
        return pdf, bin_edges
    
    else:
        raise ValueError('The method not implemented.')

def estimate_pdf_conditional(data, data_cond, val_cond, bins='fd', method='hist'):
    """Estimate the conditional probability density function of the data 
        by the histogram method.
    
    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations, n_variables)
        The data.
    data_cond : numpy.ndarray of shape (n_observations, n_conditional_variables)
        The data to be conditioned on.
    val_cond : int
        The value of the variable to condition on.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n_1, n_2, ..., n_n : The number of bins for each variable.
        - n : The number of bins for all variables.
    method : str, optional
        (default = 'hist')
        The method to estimate the probability density function (pdf).
        - 'hist' : The pdf is estimated by the histogram method.
        - 'kde' : The pdf is estimated by the kernel density estimation method.
    
    Returns
    -------
    pdf_conditional : numpy.ndarray of shape (n_bins, n_variables)
        The estimated probability density function of the data.
    x : list of numpy.ndarray of shape (n_bins-1,)
        The x values of the corresponding bins.
    
    """
    if method == 'hist':
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data_cond.ndim == 1:
            data_cond = data_cond.reshape(-1, 1)
        elif data_cond.ndim == 2:
            if data_cond.shape[1] != 1:
                raise ValueError(
                    'The data to be conditioned on must be ' \
                    + 'a 2D array of shape (n_observations, 1). '
                )
        else:
            raise ValueError(
                'The data must be a 2D array of ' \
                + 'shape (n_observations, 1). '
            )

        # Combine the data and the data to be conditioned on
        data_joint = np.hstack((data, data_cond))

        # Compute the joint probability density function
        pdf_joint, bin_edges_joint = estimate_pdf_joint(
            data=data_joint, 
            bins=bins,
            method=method
        )
        pdf_cond, _ = estimate_pdf(
            data_cond, 
            bins=bins, 
            method=method
        )

        x = bin_edges_joint[0]

        if pdf_cond[val_cond] == 0.:
            return np.zeros_like(pdf_joint[..., val_cond]), x
        else:
            return pdf_joint[..., val_cond] / pdf_cond[val_cond], x

    else:
        raise ValueError('The method not implemented.')

def pdf_evolution(X, t_max, n_x_resolution=50):
    """Estimate the time evolution of the probability density function of the data.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_nodes, n_timesteps, n_variables)
        The data.
    t_max : float
        The maximum time.
    n_x_resolution : int, optional
        (default = 50)
        The number of bins to estimate the probability density function.
    
    Returns
    -------
    time_evolution : numpy.ndarray of shape (n_nodes, n_x_resolution, n_timesteps)
        The time evolution of the probability density function.
    x_grid : numpy.ndarray of shape (n_x_resolution,)
        The x values of the corresponding bins.
    time_grid : numpy.ndarray of shape (n_timesteps,)
        The time steps.
    """
    n_nodes, n_timesteps, _ = X.shape

    X_max_amp = np.max(np.abs(X)) * 1.5
    x_grid = np.linspace(-X_max_amp, X_max_amp, n_x_resolution+1)
    time_grid = np.linspace(0, t_max, n_timesteps)

    time_evolution = np.zeros((n_nodes, n_x_resolution, n_timesteps))
    for i in range(n_nodes):
        for t in range(n_timesteps):
            p_t, _ = estimate_pdf(X[i, t, :], bins=x_grid)
            assert(1 - np.sum(p_t) < 1e-10)
            time_evolution[i, :, t] = p_t
    return time_evolution, x_grid, time_grid

def covariance(data):
    """Calculate the covariance matrix.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_nodes, n_timesteps, n_samples)
        The time series data.

    Returns
    -------
    cov_np : numpy.ndarray of shape (n_samples, n_nodes * n_nodes)
        The covariance matrix.
    
    """
    _, _, n_samples = data.shape
    cov_list = []
    for i in range(n_samples):
        X_i = data[:, :, i]
        cov_i = np.cov(X_i)
        cov_list.append(cov_i.flatten())
    cov_np = np.array(cov_list)
    return cov_np

def conditional_expectation(X, Y, Z, bins='fd'):
    """Conditional expectation.
    
    Parameter
    ---------
    X : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data.
    Y : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data.
    Z : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data. (condition)
    bins : int or str, optional
        (default = 'fd')
        The number of bins or the method to compute the number of bins.
        - 'fd' : Freedman-Diaconis rule
        - n (int) : The number of bins.

    Returns
    -------
    z_bins : numpy.ndarray of shape (n_bins,)
        The bin values of the conditional variable.
    mean : tuple of 2 numpy.ndarray of shape (n_bins,)
        The conditional expectation of node i and j
            - X1_mean : numpy.ndarray of shape (n_bins,)
                The conditional expectation of node i.
            - X2_mean : numpy.ndarray of shape (n_bins,) 
                The conditional expectation of node j.
    std : tuple of 2 numpy.ndarray of shape (n_bins,)
        The standard deviation of node i and j
            - X1_std : numpy.ndarray of shape (n_bins,)
                The conditional standard deviation of node i.
            - X2_std : numpy.ndarray of shape (n_bins,)
                The conditional standard deviation of node j.
    X3_dig : numpy.ndarray of shape (n_samples,)
        The digitised data of node v3.
    
    """
    # Check the shape of the data
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            'The shapes of X, Y, and Z must be the same.'
        )

    # Get the number of bins based on Freedman-Diaconis rule or user input
    if isinstance(bins, str) and bins == 'fd':
        z_bins = freedman_diaconis_rule(Z.flatten(), trim=Z.shape[-1])
        n_bins = len(z_bins) - 1
    elif isinstance(bins, int):
        max_amp = np.max(np.abs(Z))
        _bins = np.linspace(-max_amp, max_amp, bins+1)
        z_bins = np.histogram_bin_edges(Z, bins=_bins)
        n_bins = bins
    else:
        raise ValueError(
            'The number of bins must be an integer or string.'
        )
    
    # Get the digitised data
    Z_dig = np.digitize(Z, z_bins)

    # Initialise the conditional expectations and standard deviations
    X_mean = np.zeros(n_bins)
    Y_mean = np.zeros(n_bins)
    X_std = np.zeros(n_bins)
    Y_std = np.zeros(n_bins)

    # Loop over bins
    for i in range(n_bins):
        if np.sum(Z_dig == i) == 0:
            X_mean[i] = np.nan
            Y_mean[i] = np.nan
            X_std[i] = np.nan
            Y_std[i] = np.nan
        else:
            X_mean[i] = np.mean(X[Z_dig == i])
            Y_mean[i] = np.mean(Y[Z_dig == i])
            X_std[i] = np.std(X[Z_dig == i])
            Y_std[i] = np.std(Y[Z_dig == i])

    # Get the bin values
    z = 0.5 * (z_bins[1:] + z_bins[:-1])
        
    return z, (X_mean, Y_mean), (X_std, Y_std), Z_dig

def conditional_variance(X, Z, bins='fd'):
    """Calculate the conditional variance.
    
    Parameter
    ---------
    X : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data.
    Z : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data. (condition)
    bins : int or str, optional
        (default = 'fd')
        The number of bins or the method to compute the number of bins.
        - 'fd' : Freedman-Diaconis rule
    
    Returns
    -------
    var_cond : numpy.ndarray of shape (n_bins,)
        The conditional variance.
    z : numpy.ndarray of shape (n_bins,)
        The bin values of the conditional variable.
    
    """
    # Check the shape of the data
    if X.shape != Z.shape:
        raise ValueError(
            'The shape of X and Z must be the same.'
        )
    
    # Get the number of bins based on Freedman-Diaconis rule or user input
    if isinstance(bins, str) and bins == 'fd':
        _bins = freedman_diaconis_rule(Z)
        n_bins = len(_bins)
    elif isinstance(bins, int):
        max_amp = np.max(np.abs(Z))
        _bins = np.linspace(-max_amp, max_amp, bins+1)
        n_bins = bins
    else:
        raise ValueError(
            'The number of bins must be an integer or string.'
        )

    # Get the bin edges
    z_bins = np.histogram_bin_edges(Z, bins=_bins)
    
    # Get the digitised data
    Z_dig = np.digitize(Z, z_bins)

    # Initialise the conditional variance
    var_cond = np.zeros(n_bins)

    # Loop over bins
    for i in range(n_bins):
        if np.sum(Z_dig == i) < 2:
            var_cond[i] = np.nan
        else:
            var_cond[i] = np.var(X[Z_dig == i])
    
    # Get the z values corresponding to each bin
    z = 0.5 * (z_bins[1:] + z_bins[:-1])

    return var_cond, z

def conditional_covariance(X, Y, Z, bins='fd'):
    """Compute the conditional variance.
    
    Parameter
    ---------
    X : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data.
    Y : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data. 
    Z : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data. (condition)
    bins : int or str, optional
        (default = 'fd')
        The number of bins or the method to compute the number of bins.
        - 'fd' : Freedman-Diaconis rule
    
    Returns
    -------
    cov_cond : numpy.ndarray of shape (n_bins,)
        The conditional covariance.
    z : numpy.ndarray of shape (n_bins,)
        The bin values of the conditional variable.
    
    """
    # Check the shape of the data
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            'The shape of X, Y, and Z must be the same.'
        )
    
    # Get the number of bins based on Freedman-Diaconis rule or user input
    if isinstance(bins, str) and bins == 'fd':
        _bins = freedman_diaconis_rule(Z)
        n_bins = len(_bins)
    elif isinstance(bins, int):
        max_amp = np.max(np.abs(Z))
        _bins = np.linspace(-max_amp, max_amp, bins+1)
        n_bins = bins
    else:
        raise ValueError(
            'The number of bins must be an integer or string.'
        )
    
    # Get the digitised data
    Z_dig = np.digitize(Z, _bins)

    # Initialise the conditional variance
    cov_cond = np.zeros(n_bins)

    # Loop over bins
    for i in range(n_bins):
        if np.sum(Z_dig == i) < 2:
            cov_cond[i] = np.nan
        else:
            cov_cond[i] = np.cov([X[Z_dig == i], Y[Z_dig == i]])[0, 1]
    
    # Get the z values corresponding to each bin
    z = 0.5 * (_bins[1:] + _bins[:-1])

    return cov_cond, z

def conditional_correlation(X, Y, Z, bins='fd', method='default'):
    """Compute the conditional variance.
    
    Parameter
    ---------
    X : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data.
    Y : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data. 
    Z : numpy.ndarray of shape (n_timesteps, n_samples)
        The time series data. (condition)
    bins : int or str, optional
        (default = 'fd')
        The number of bins or the method to compute the number of bins.
        - 'fd' : Freedman-Diaconis rule
    method : str, optional
        (default = 'default')
        The method to compute the conditional correlation.
        - 'default' : Pearson correlation coefficient
        - 'manual' : manual computation
    
    Returns
    -------
    corr_cond : numpy.ndarray of shape (n_bins,)
        The conditional correlation.
    z : numpy.ndarray of shape (n_bins,)
        The bin values of the conditional variable.
    corr_cond_err : numpy.ndarray of shape (n_bins,)
        The standard error of the conditional correlation.
    
    """
    # Check the shape of the data
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            'The shape of X, Y, and Z must be the same.'
        )
    if Z.ndim == 1 or (Z.ndim == 2 and Z.shape[1] == 1):
        # Get the number of bins based on Freedman-Diaconis rule or user input
        if isinstance(bins, str) and bins == 'fd':
            _bins = freedman_diaconis_rule(Z)
            n_bins = len(_bins) - 1
        elif isinstance(bins, int):
            max_amp = np.max(np.abs(Z))
            _bins = np.linspace(-max_amp, max_amp, bins+1)
            n_bins = len(_bins) - 1
        else:
            raise ValueError(
                'The number of bins must be an integer or string.'
            )

        # Initialise the conditional correlation
        cond_corr = np.zeros(n_bins)
        cond_corr_stderr = np.zeros(n_bins)
        
        # Get the digitised data
        Z_dig = np.digitize(Z, _bins)
        X = np.digitize(X, freedman_diaconis_rule(X))
        Y = np.digitize(Y, freedman_diaconis_rule(Y))

        if method == 'default':
            # Loop over bins
            for j in range(n_bins):
                if np.sum(Z_dig == j) < 10:
                    cond_corr[j] = np.nan
                else:
                    cond_corr[j] = np.corrcoef(X[Z_dig == j], Y[Z_dig == j])[0, 1]
                    cond_corr_stderr[j] = np.sqrt((1 - cond_corr[j]**2) / (np.sum(Z_dig == j) - 2))
                    # cond_corr[j] = np.cov(X[Z_dig == j], Y[Z_dig == j])[0, 1] / np.sqrt(np.var(X[Z_dig == j], ddof=1) * np.var(Y[Z_dig == j], ddof=1))
        elif method == 'manual':
            cov, _ = conditional_covariance(X, Y, Z, bins=bins)
            var_X, _ = conditional_variance(X, Z, bins=bins)
            var_Y, _ = conditional_variance(Y, Z, bins=bins)

            cond_corr = cov / np.sqrt(var_X * var_Y)
            cond_corr_stderr = np.sqrt((1 - cond_corr**2) / (np.sum(Z_dig == j) - 2))
        else:
            raise ValueError(
                'The method not implemented.'
            )
        
        z = 0.5 * (_bins[1:] + _bins[:-1])
        return cond_corr, z, cond_corr_stderr
    
    elif Z.ndim == 2 and Z.shape[1] > 1:
        # Get the number of samples
        _, n_samples = Z.shape
        
        # Get the number of bins based on Freedman-Diaconis rule or user input
        if isinstance(bins, str) and bins == 'fd':
            _bins = freedman_diaconis_rule(Z.flatten(), trim=n_samples)
            n_bins = len(_bins) - 1
        elif isinstance(bins, int):
            _bins = np.linspace(-np.max(np.abs(Z)), np.max(np.abs(Z)), bins+1)
            n_bins = len(_bins) - 1
        else:
            raise ValueError(
                'The number of bins must be an integer or string.'
            )

        # Initialise the conditional correlation
        cond_corr = np.zeros((n_bins, n_samples))
        cond_corr_stderr = np.zeros((n_bins, n_samples))
        
        # Loop over samples
        for i in range(n_samples):
            # Get the digitised data
            Z_dig_i = np.digitize(Z[:, i], _bins)

            if method == 'default':
                # Loop over bins
                for j in range(n_bins):
                    if np.sum(Z_dig_i == j) < 10:
                        cond_corr[j, i] = np.nan
                    else:
                        cond_corr[j, i] = np.corrcoef(X[:, i][Z_dig_i == j], Y[:, i][Z_dig_i == j])[0, 1]
                        cond_corr_stderr[j, i] = np.sqrt((1 - cond_corr[j, i]**2) / (np.sum(Z_dig_i == j) - 2))
                        # cond_corr[j, i] = np.cov(X[:, i][Z_dig_i == j], Y[:, i][Z_dig_i == j])[0, 1] / np.sqrt(np.var(X[:, i][Z_dig_i == j], ddof=1) * np.var(Y[:, i][Z_dig_i == j], ddof=1))

            elif method == 'manual':
                cov, _ = conditional_covariance(X[:, i], Y[:, i], Z[:, i], bins=n_bins)
                var_X, _ = conditional_variance(X[:, i], Z[:, i], bins=n_bins)
                var_Y, _ = conditional_variance(Y[:, i], Z[:, i], bins=n_bins)

                cond_corr[:, i] = cov / np.sqrt(var_X * var_Y)
                cond_corr_stderr[:, i] = np.sqrt((1 - cond_corr[:, i]**2) / (np.sum(Z_dig_i == j) - 2))
            else:
                raise ValueError(
                    'The method not implemented.'
                )
        
        z = 0.5 * (_bins[1:] + _bins[:-1])
        return cond_corr, z, cond_corr_stderr

def entropy(pdf, x):
    """Calculate the entropy of the probability density function.

    Parameters
    ----------
    pdf : numpy.ndarray of shape (n_bins,)
        The probability density function.
    x : numpy.ndarray of shape (n_bins,)
        The x values of the corresponding bins.
    
    Returns
    -------
    entropy : float
        The entropy of the probability density function.
    
    """
    # Calculate the bin width
    dx = x[1] - x[0]
    assert(dx > 0.)

    # Calculate the entropy
    return - np.dot(pdf[pdf > 0.] * dx, np.log(pdf[pdf > 0.] * dx))
    # return - np.dot(pdf[pdf > 0.], np.log(pdf[pdf > 0.]))

def entropy_joint(pdf_joint, x):
    """Calculate the joint entropy of the probability density function.

    Parameters
    ----------
    pdf_joint : numpy.ndarray of shape (n_bins, n_variables)
        The joint probability density function.
    x : list of numpy.ndarray of shape (n_bins,)
        The x values of the corresponding bins.
    
    Returns
    -------
    entropy_joint : float
        The joint entropy of the probability density function.
    
    """
    # Calculate the bin width
    dx = np.array([x[d][1] - x[d][0] for d in range(len(x))])

    # Calculate the volume of each bin
    dV = np.prod(dx)
    assert(dV > 0.)

    # Calculate the joint entropy
    return - np.dot(pdf_joint[pdf_joint > 0.] * dV, np.log(pdf_joint[pdf_joint > 0.] * dV))
    # return - np.dot(pdf_joint[pdf_joint > 0.], np.log(pdf_joint[pdf_joint > 0.]))

def conditional_mutual_information(X, Y, Z, bins='fd', method='hist'):
    """Calculate the conditional mutual information between X and Y given Z.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_observations, )
        The data.
    Y : numpy.ndarray of shape (n_observations, )
        The data.
    Z : numpy.ndarray of shape (n_observations, )
        The data to be conditioned.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n_1, n_2, ..., n_n : The number of bins for each variable.
        - n : The number of bins for all variables.
    method : str, optional
        (default = 'hist')
        The method to estimate the probability density function.
        - 'hist' : The pdf is estimated by the histogram method.
        - 'kde' : The pdf is estimated by the kernel density estimation method.
    
    Returns
    -------
    cmi : numpy.ndarray of shape (n_bins, n_samples)
        The conditional mutual information between X and Y given Z=z for each z in Z.
    z : numpy.ndarray of shape (n_bins,)
        The z values of the corresponding bins.
    
    """
    # Check the shape of the data
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            'The shape of X, Y, and Z must be the same.'
        )
    if Z.ndim == 1 or (Z.ndim == 2 and Z.shape[1] == 1):
        # Get the number of bins based on Freedman-Diaconis rule or user input
        if isinstance(bins, str) and bins == 'fd':
            _bins = freedman_diaconis_rule(Z)
            n_bins = len(_bins) - 1
        elif isinstance(bins, int):
            max_amp = np.max(np.abs(Z))
            _bins = np.linspace(-max_amp, max_amp, bins+1)
            n_bins = len(_bins) - 1
        else:
            raise ValueError(
                'The number of bins must be an integer or string.'
            )

        cmi = np.zeros(n_bins)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        Z_dig = np.digitize(Z, _bins)
        
        for i in range(n_bins):
            if np.sum(Z_dig == i) == 0:
                cmi[i] = np.nan
                continue
            pdf_X, _x = estimate_pdf(X[Z_dig == i], bins=bins, method=method)
            pdf_Y, _y = estimate_pdf(Y[Z_dig == i], bins=bins, method=method)
            pdf_XY, _xy = estimate_pdf_joint(np.vstack((X[Z_dig == i], Y[Z_dig == i])).T, bins=bins, method=method)
            H_X = entropy(pdf_X, _x)
            H_Y = entropy(pdf_Y, _y)
            H_XY = entropy_joint(pdf_XY, _xy)
            # assert(H_X + H_Y >= H_XY)
            cmi[i] = H_X + H_Y - H_XY
        
        z = 0.5 * (_bins[1:] + _bins[:-1])

        return cmi, z
    
    elif Z.ndim == 2 and Z.shape[1] > 1:
        _, n_samples = Z.shape
        
        # Get the number of bins based on Freedman-Diaconis rule or user input
        if isinstance(bins, str) and bins == 'fd':
            _bins = freedman_diaconis_rule(Z.flatten(), trim=n_samples)
            n_bins = len(_bins) - 1
        elif isinstance(bins, int):
            max_amp = np.max(np.abs(Z))
            _bins = np.linspace(-max_amp, max_amp, bins+1)
            n_bins = len(_bins) - 1
        else:
            raise ValueError(
                'The number of bins must be an integer or string.'
            )

        cmi = np.zeros((n_bins, n_samples))

        for j in range(n_samples):
            X_j = X[:, j].reshape(-1, 1)
            Y_j = Y[:, j].reshape(-1, 1)
            Z_j = Z[:, j].reshape(-1, 1)
            
            Z_dig = np.digitize(Z_j, _bins)
            
            for i in range(n_bins):
                pdf_X, _x = estimate_pdf(X_j[Z_dig == i], bins=bins, method=method)
                pdf_Y, _y = estimate_pdf(Y_j[Z_dig == i], bins=bins, method=method)
                pdf_XY, _xy = estimate_pdf_joint(np.vstack((X_j[Z_dig == i], Y_j[Z_dig == i])).T, bins=bins, method=method)
                H_X = entropy(pdf_X[:, i], _x)
                H_Y = entropy(pdf_Y[:, i], _y)
                H_XY = entropy_joint(pdf_XY[:, :, i], _xy)
                # assert(H_X + H_Y >= H_XY)
                cmi[i, j] = H_X + H_Y - H_XY

        z = 0.5 * (_bins[1:] + _bins[:-1])

        return cmi, z

"""End of file."""