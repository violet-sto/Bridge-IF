import torch
from torch.nn import functional as F
import numpy as np
import math

# from src.data.utils import PlaceHolder


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def sample_gaussian(size):
    x = torch.randn(size)
    return x


def sample_gaussian_with_mask(size, node_mask):
    x = torch.randn(size)
    x = x.type_as(node_mask.float())
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


# def cosine_beta_schedule_discrete(timesteps, s=0.008):
#     """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
#     steps = timesteps + 2
#     x = np.linspace(0, steps, steps)

#     alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas = 1 - alphas
#     return betas.squeeze()


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    betas = np.insert(betas, 0, 0)
    return betas.squeeze()    


def cosineSD_beta_schedule_discrete(timesteps, s=0.008, d=9):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)

    alphas_cumprod = 1 / (1 + (d * np.tan(0.5 * np.pi * ((x / steps) + s) / (1 + s))) ** 2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = 1 - alphas_cumprod[::-1] # reverse timestep
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    betas = np.insert(betas, 0, 0)
    return betas.squeeze()  


# def interpolation_beta_schedule_discrete(timesteps):
#     """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
#     steps = timesteps + 2
#     x = np.linspace(0, steps, steps)

#     alphas_cumprod = ((steps - x) / steps)
#     alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas = 1 - alphas
#     return betas.squeeze()    


def interpolation_beta_schedule_discrete(timesteps):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)

    alphas_cumprod = ((steps - x) / steps)
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    betas = np.insert(betas, 0, 0)
    return betas.squeeze()   


# def polynomial_beta_schedule_discrete(timesteps, s=0.008):
#     steps = timesteps + 2
#     x = np.linspace(0, steps, steps)
#     alphas = (1 - 2 * s) * (1 - (x / steps) ** 2)[1:]
#     betas = 1 - alphas
#     return betas.squeeze()


def polynomial_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas = (1 - 2 * s) * (1 - (x / steps) ** 2)[1:]
    betas = 1 - alphas
    betas = np.insert(betas, 0, 0)
    return betas.squeeze()


# def linear_beta_schedule_discrete(timesteps, s=0.008):
#     steps = timesteps + 2
#     x = np.linspace(0, steps, steps)
#     alphas = (1 - 2 * s) * (1 - (x / steps))[1:]
#     betas = 1 - alphas
#     return betas.squeeze()


def linear_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas = (1 - 2 * s) * (1 - (x / steps))[1:]
    betas = 1 - alphas
    betas = np.insert(betas, 0, 0)
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)



def gaussian_KL(q_mu, q_sigma):
    """Computes the KL distance between a normal distribution and the standard normal.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch((torch.log(1 / q_sigma) + 0.5 * (q_sigma ** 2 + q_mu ** 2) - 0.5))


def cdf_std_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def sigma_and_alpha_t_given_s(gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size):
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_feature_noise(X_size, E_size, y_size, node_mask):
    """Standard normal noise for all features.
        Output size: X.size(), E.size(), y.size() """
    # TODO: How to change this for the multi-gpu case?
    epsX = sample_gaussian(X_size)
    epsE = sample_gaussian(E_size)
    epsy = sample_gaussian(y_size)

    float_mask = node_mask.float()
    epsX = epsX.type_as(float_mask)
    epsE = epsE.type_as(float_mask)
    epsy = epsy.type_as(float_mask)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(epsE)
    indices = torch.triu_indices(row=epsE.size(1), col=epsE.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    epsE = epsE * upper_triangular_mask
    epsE = (epsE + torch.transpose(epsE, 1, 2))

    assert (epsE == torch.transpose(epsE, 1, 2)).all()

    return PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)


def sample_normal(mu_X, mu_E, mu_y, sigma, node_mask):
    """Samples from a Normal distribution."""
    # TODO: change for multi-gpu case
    eps = sample_feature_noise(mu_X.size(), mu_E.size(), mu_y.size(), node_mask).type_as(mu_X)
    X = mu_X + sigma * eps.X
    E = mu_E + sigma.unsqueeze(1) * eps.E
    y = mu_y + sigma.squeeze(1) * eps.y
    return PlaceHolder(X=X, E=E, y=y)


def check_issues_norm_values(gamma, norm_val1, norm_val2, num_stdevs=8):
    """ Check if 1 / norm_value is still larger than 10 * standard deviation. """
    zeros = torch.zeros((1, 1))
    gamma_0 = gamma(zeros)
    sigma_0 = sigma(gamma_0, target_shape=zeros.size()).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sigma_0 * num_stdevs > 1. / max_norm_value:
        raise ValueError(
            f'Value for normalization value {max_norm_value} probably too '
            f'large with sigma_0 {sigma_0:.5f} and '
            f'1 / norm_value = {1. / max_norm_value}')


def sample_discrete_features(probX):
    ''' Sample features from multinomial distribution with given probabilities probX
        :param probX: bs, n, dx_out        node features
    '''
    bs, n, _ = probX.shape
    # Noise X
    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)              # (bs * n, 1)
    X_t = X_t.reshape(bs, n)                # (bs, n)

    return X_t


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)    # (bs, N, d)

    return prob


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    # Add a small value everywhere to avoid nans
    pred_X = pred_X + 1e-7
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)

    pred_E = pred_E + 1e-7
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)


    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def cbo0pdi_X(X_t, Qt, Qsb, Qtb):
    """ Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: (bs, n, dt)
        Qt: (bs, n, d_t-1, dt)
        Qsb: (bs, n, d0, d_t-1)
        Qtb: (bs, n, d_0, dt)
    """

    Qt_T = Qt.transpose(-1, -2)  # bs, n, dt, d_t-1
    left_term = X_t.unsqueeze(-2) @ Qt_T  # bs, n, 1, d_t-1
    numerator = left_term * Qsb  # bs, n, d0, d_t-1
    denominator = Qtb @ X_t.unsqueeze(-1)  # bs, n, d_0, 1
    denominator[denominator == 0] = 1e-6
    out = numerator / denominator  # bs, n, d0, d_t-1

    return out


def cbo0pdi_E(E_t, Qt, Qsb, Qtb):
    """ Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        E_t: (bs, n, n, dt)
        Qt: (bs, n, n, d_t-1, dt)
        Qsb: (bs, n, n, d0, d_t-1)
        Qtb: (bs, n, n, d_0, dt)
    """
    E_t = E_t.flatten(start_dim=1, end_dim=2).to(torch.float32)  # bs, N, dt
    Qt = Qt.flatten(start_dim=1, end_dim=2).to(torch.float32)  # bs, N, d_t-1, dt
    Qsb = Qsb.flatten(start_dim=1, end_dim=2).to(torch.float32)  # bs, N, d0, d_t-1
    Qtb = Qtb.flatten(start_dim=1, end_dim=2).to(torch.float32)  # bs, N, d_0, dt

    Qt_T = Qt.transpose(-1, -2)  # bs, N, dt, d_t-1
    left_term = E_t.unsqueeze(-2) @ Qt_T  # bs, N, 1, d_t-1
    numerator = left_term * Qsb  # bs, N, d0, d_t-1
    denominator = Qtb @ E_t.unsqueeze(-1)  # bs, N, d_0, 1
    denominator[denominator == 0] = 1e-6
    out = numerator / denominator  # bs, N, d0, d_t-1

    return out
