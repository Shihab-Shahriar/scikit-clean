# Code is taken from https://github.com/cgnorthcutt/cleanlab

import numpy as np
from sklearn.utils import check_random_state


def noise_matrix_is_valid(noise_matrix, py, verbose=False):
    '''Given a prior py = p(y=k), returns true if the given noise_matrix is a learnable matrix.
    Learnability means that it is possible to achieve better than random performance, on average,
    for the amount of noise in noise_matrix.'''

    # Number of classes
    K = len(py)

    # Let's assume some number of training examples for code readability,
    # but it doesn't matter what we choose as its not actually used.
    N = float(10000)

    ps = np.dot(noise_matrix, py)  # P(y=k)

    # P(s=k, y=k')
    joint_noise = np.multiply(noise_matrix, py)  # / float(N)

    # Check that joint_probs is valid probability matrix
    if not (abs(joint_noise.sum() - 1.0) < 1e-6):
        return False

    # Check that noise_matrix is a valid matrix
    # i.e. check p(s=k)*p(y=k) < p(s=k, y=k)
    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print(
                "E1E2/C", round(E1 * E2 / C),
                "E1", round(E1),
                "E2", round(E2),
                "C", round(C),
                "|", round(E1 * E2 / C + E1 + E2 + C),
                "|", round(E1 * E2 / C), "<", round(O),
            )
            print(
                round(ps[i] * py[i]), "<", round(joint_noise[i][i]),
                ":", ps[i] * py[i] < joint_noise[i][i],
            )

        if not (ps[i] * py[i] < joint_noise[i][i]):
            return False

    return True


def generate_n_rand_probabilities_that_sum_to_m(
        n,
        m,
        seed,
        max_prob=1.0,
        min_prob=0.0,
):
    '''When min_prob=0 and max_prob = 1.0, this method is deprecated.
    Instead use np.random.dirichlet(np.ones(n))*m

    Generates 'n' random probabilities that sum to 'm'.

    Parameters
    ----------

    n : int
      Length of np.array of random probabilities to be returned.

    m : float
      Sum of np.array of random probabilites that is returned.

    max_prob : float (0.0, 1.0] | Default value is 1.0
      Maximum probability of any entry in the returned np.array.

    min_prob : float [0.0, 1.0) | Default value is 0.0
      Minimum probability of any entry in the returned np.array.'''

    epsilon = 1e-6  # Imprecision allowed for inequalities with floats
    rns = check_random_state(seed)

    if n == 0:
        return np.array([])
    if (max_prob + epsilon) < m / float(n):
        raise ValueError("max_prob must be greater or equal to m / n, but " +
                         "max_prob = " + str(max_prob) + ", m = " + str(m) + ", n = " +
                         str(n) + ", m / n = " + str(m / float(n)))
    if min_prob > (m + epsilon) / float(n):
        raise ValueError("min_prob must be less or equal to m / n, but " +
                         "max_prob = " + str(max_prob) + ", m = " + str(m) + ", n = " +
                         str(n) + ", m / n = " + str(m / float(n)))

    # When max_prob = 1, min_prob = 0, the following two lines are equivalent to:
    #   intermediate = np.sort(np.append(np.random.uniform(0, 1, n-1), [0, 1]))
    #   result = (intermediate[1:] - intermediate[:-1]) * m
    result = rns.dirichlet(np.ones(n)) * m

    min_val = min(result)
    max_val = max(result)
    while max_val > (max_prob + epsilon):
        new_min = min_val + (max_val - max_prob)
        # This adjustment prevents the new max from always being max_prob.
        adjustment = (max_prob - new_min) * rns.rand()
        result[np.argmin(result)] = new_min + adjustment
        result[np.argmax(result)] = max_prob - adjustment
        min_val = min(result)
        max_val = max(result)

    min_val = min(result)
    max_val = max(result)
    while min_val < (min_prob - epsilon):
        min_val = min(result)
        max_val = max(result)
        new_max = max_val - (min_prob - min_val)
        # This adjustment prevents the new min from always being min_prob.
        adjustment = (new_max - min_prob) * rns.rand()
        result[np.argmax(result)] = new_max - adjustment
        result[np.argmin(result)] = min_prob + adjustment
        min_val = min(result)
        max_val = max(result)

    return result


def randomly_distribute_N_balls_into_K_bins(
        N,  # int
        K,  # int
        seed,
        max_balls_per_bin=None,
        min_balls_per_bin=None,
):
    '''Returns a uniformly random numpy integer array of length N that sums to K.'''

    if N == 0:
        return np.zeros(K, dtype=int)
    if max_balls_per_bin is None:
        max_balls_per_bin = N
    else:
        max_balls_per_bin = min(max_balls_per_bin, N)
    if min_balls_per_bin is None:
        min_balls_per_bin = 0
    else:
        min_balls_per_bin = min(min_balls_per_bin, N / K)
    if N / float(K) > max_balls_per_bin:
        N = max_balls_per_bin * K

    arr = np.round(generate_n_rand_probabilities_that_sum_to_m(
        n=K,
        m=1,
        max_prob=max_balls_per_bin / float(N),
        min_prob=min_balls_per_bin / float(N),
        seed=seed
    ) * N)
    while sum(arr) != N:
        while sum(arr) > N:  # pragma: no cover
            arr[np.argmax(arr)] -= 1
        while sum(arr) < N:
            arr[np.argmin(arr)] += 1
    return arr.astype(int)


# This can be quite slow
def generate_noise_matrix_from_trace(
        K,
        trace,
        max_trace_prob=1.0,
        min_trace_prob=1e-5,
        max_noise_rate=1 - 1e-5,
        min_noise_rate=0.0,
        valid_noise_matrix=True,
        py=None,
        frac_zero_noise_rates=0.,
        seed=0,
        max_iter=10000,
):
    '''Generates a K x K noise matrix P(s=k_s|y=k_y) with trace
    as the np.mean(np.diagonal(noise_matrix)).

    Parameters
    ----------

    K : int
      Creates a noise matrix of shape (K, K). Implies there are
      K classes for learning with noisy labels.

    trace : float (0.0, 1.0]
      Sum of diagonal entries of np.array of random probabilites that is returned.

    max_trace_prob : float (0.0, 1.0]
      Maximum probability of any entry in the trace of the return matrix.

    min_trace_prob : float [0.0, 1.0)
      Minimum probability of any entry in the trace of the return matrix.

    max_noise_rate : float (0.0, 1.0]
      Maximum noise_rate (non-digonal entry) in the returned np.array.

    min_noise_rate : float [0.0, 1.0)
      Minimum noise_rate (non-digonal entry) in the returned np.array.

    valid_noise_matrix : bool
      If True, returns a matrix having all necessary conditions for
      learning with noisy labels. In particular, p(y=k)p(s=k) < p(y=k,s=k)
      is satisfied. This requires that Trace > 1.

    py : np.array (shape (K, 1))
      The fraction (prior probability) of each true, hidden class label, P(y = k).
      REQUIRED when valid_noise_matrix == True.

    frac_zero_noise_rates : float
      The fraction of the n*(n-1) noise rates that will be set to 0. Note that if
      you set a high trace, it may be impossible to also have a low
      fraction of zero noise rates without forcing all non-"1" diagonal values.
      Instead, when this happens we only guarantee to produce a noise matrix with
      frac_zero_noise_rates **or higher**. The opposite occurs with a small trace.

    seed : int
      Seeds the random number generator for numpy.

    max_iter : int (default: 10000)
      The max number of tries to produce a valid matrix before returning False.

    Output
    ------
    np.array (shape (K, K))
      noise matrix P(s=k_s|y=k_y) with trace
      as the np.sum(np.diagonal(noise_matrix)).
      This a conditional probability matrix and a
      left stochastic matrix.'''

    if valid_noise_matrix and trace <= 1:
        raise ValueError("trace = {}. trace > 1 is necessary for a".format(trace) +
                         " valid noise matrix to be returned (valid_noise_matrix == True)")

    if valid_noise_matrix and py is None and K > 2:
        raise ValueError("py must be provided (not None) if the input parameter" +
                         " valid_noise_matrix == True")

    if K <= 1:
        raise ValueError('K must be >= 2, but K = {}.'.format(K))

    if max_iter < 1:
        return False

    rns = check_random_state(seed)

    # Special (highly constrained) case with faster solution.
    # Every 2 x 2 noise matrix with trace > 1 is valid because p(y) is not used
    if K == 2:
        if frac_zero_noise_rates >= 0.5:  # Include a single zero noise rate
            noise_mat = np.array([
                [1., 1 - (trace - 1.)],
                [0., trace - 1.],
            ])
            return noise_mat if rns.rand() > 0.5 else np.rot90(noise_mat, k=2)
        else:  # No zero noise rates
            diag = generate_n_rand_probabilities_that_sum_to_m(2, trace, seed=rns.randint(100))
            noise_matrix = np.array([
                [diag[0], 1 - diag[1]],
                [1 - diag[0], diag[1]],
            ])
            return noise_matrix

            # K > 2
    for z in range(max_iter):
        noise_matrix = np.zeros(shape=(K, K))

        # Randomly generate noise_matrix diagonal.
        nm_diagonal = generate_n_rand_probabilities_that_sum_to_m(
            n=K,
            m=trace,
            max_prob=max_trace_prob,
            min_prob=min_trace_prob,
            seed=rns.randint(100)
        )
        np.fill_diagonal(noise_matrix, nm_diagonal)

        # Randomly distribute number of zero-noise-rates across columns
        num_col_with_noise = K - np.count_nonzero(1 == nm_diagonal)
        num_zero_noise_rates = int(K * (K - 1) * frac_zero_noise_rates)
        # Remove zeros already in [1,0,..,0] columns
        num_zero_noise_rates -= (K - num_col_with_noise) * (K - 1)
        num_zero_noise_rates = np.maximum(num_zero_noise_rates, 0)  # Prevent negative
        num_zero_noise_rates_per_col = randomly_distribute_N_balls_into_K_bins(
            N=num_zero_noise_rates,
            K=num_col_with_noise,
            max_balls_per_bin=K - 2,  # 2 = one for diagonal, and one to sum to 1
            min_balls_per_bin=0,
            seed=rns.randint(100)
        ) if K > 2 else np.array([0, 0])  # Special case when K == 2
        stack_nonzero_noise_rates_per_col = list(K - 1 - num_zero_noise_rates_per_col)[::-1]
        # Randomly generate noise rates for columns with noise.
        for col in np.arange(K)[nm_diagonal != 1]:
            num_noise = stack_nonzero_noise_rates_per_col.pop()
            # Generate num_noise noise_rates for the given column.
            noise_rates_col = list(generate_n_rand_probabilities_that_sum_to_m(
                n=num_noise,
                m=1 - nm_diagonal[col],
                max_prob=max_noise_rate,
                min_prob=min_noise_rate,
                seed=rns.randint(100),
            ))
            # Randomly select which rows of the noisy column to assign the random noise rates
            rows = rns.choice([row for row in range(K) if row != col], num_noise, replace=False)
            for row in rows:
                noise_matrix[row][col] = noise_rates_col.pop()
        if not valid_noise_matrix or noise_matrix_is_valid(noise_matrix, py):
            break

    return noise_matrix


def gen_simple_noise_mat(K: int, noise_level: float, random_state=None):
    rns = check_random_state(random_state)

    mat = np.zeros((K, K), dtype='float')

    mean = 1 - noise_level
    diag = rns.normal(loc=mean, scale=mean / 10, size=5)
    np.fill_diagonal(mat, diag)
    print(diag)

    for i in range(K):
        nl = 1 - mat[i][i]
        cols = [j for j in range(K) if j != i]
        mat[i, cols] = rns.dirichlet(np.ones(K - 1)) * nl

    return mat



