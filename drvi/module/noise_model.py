import torch
from scvi.distributions import NegativeBinomial
from torch.distributions import Distribution, Normal, Poisson, LogNormal
from torch.nn import functional as F


class NoiseModel:
    def __init__(self, aux_prefix='__nm_'):
        self.aux_prefix = aux_prefix

    @property
    def parameters(self):
        raise NotImplementedError()

    @property
    def main_param(self):
        return 'mean'

    def initial_transformation(self, x, x_mask=1.):
        x = x
        aux_info = {}
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        # lib_y is so strange here. I should think about it.
        raise NotImplementedError()


class NormalNoiseModel(NoiseModel):
    def __init__(self, model_var='fixed', eps=1e-8, aux_prefix='__nm_'):
        super().__init__(aux_prefix=aux_prefix)
        self.model_var = model_var
        self.eps = eps

    @property
    def parameters(self):
        if self.model_var == 'fixed':
            var_desc = 'fixed=1e-2'
        elif self.model_var.startswith('fixed='):
            var_desc = self.model_var
        elif self.model_var == 'dynamic':
            var_desc = 'no_transformation'
        elif self.model_var == 'feature':
            var_desc = 'per_feature'
        else:
            raise NotImplementedError()
        return {
            'mean': 'no_transformation',
            'var': var_desc,
        }

    def initial_transformation(self, x, x_mask=1.):
        x = x
        aux_info = {}
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        var = parameters['var']
        if self.model_var:
            var = torch.nan_to_num(torch.exp(var), posinf=100, neginf=0) + self.eps
        return Normal(parameters['mean'], torch.abs(var).sqrt())


class LogNormalNoiseModel(NoiseModel):
    def __init__(self, model_var=False, aux_prefix='__nm_'):
        super().__init__(aux_prefix=aux_prefix)
        self.model_var = model_var

    @property
    def parameters(self):
        return {
            'mean': 'no_transformation',
            'var': 'no_transformation' if self.model_var else 'fixed=1e-2',
        }

    def initial_transformation(self, x, x_mask=1.):
        x = x
        aux_info = {}
        aux_info[f'{self.aux_prefix}x_library_size'] = (x_mask * x).sum(dim=-1) if x_mask is not None else x.sum(dim=-1)
        x = torch.log1p(x / aux_info[f'{self.aux_prefix}x_library_size'].unsqueeze(-1) * 1e4)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        mean = parameters['mean']
        var = torch.abs(parameters['var'])
        library_size = lib_y

        trans_mean = torch.expm1(mean)
        trans_mean = torch.where(trans_mean >= 0, library_size.unsqueeze(-1) / 1e4 * trans_mean, trans_mean / 2)
        trans_mean = torch.log1p(trans_mean)
        return LogNormal(trans_mean, var.sqrt())


class PoissonNoiseModel(NoiseModel):
    def __init__(self, aux_prefix='__nm_'):
        super().__init__(aux_prefix=aux_prefix)

    @property
    def parameters(self):
        return {
            'mean': 'no_transformation',
        }

    def initial_transformation(self, x, x_mask=1.):
        x = x
        aux_info = {}
        aux_info[f'{self.aux_prefix}x_library_size'] = (x_mask * x).sum(dim=-1) if x_mask is not None else x.sum(dim=-1)
        x = torch.log1p(x / aux_info[f'{self.aux_prefix}x_library_size'].unsqueeze(-1) * 1e4)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        mean = parameters['mean']
        library_size = lib_y

        trans_mean = torch.exp(mean)
        trans_mean = library_size.unsqueeze(-1) / 1e4 * trans_mean
        return Poisson(trans_mean)


class NegativeBinomialNoiseModel(NoiseModel):
    def __init__(self, dispersion='feature', aux_prefix='__nm_', mean_transformation='exp'):
        super().__init__(aux_prefix=aux_prefix)
        assert mean_transformation in ['exp', 'softmax']
        self.dispersion = dispersion
        self.mean_transformation = mean_transformation

    @property
    def parameters(self):
        params = {
            'mean': 'no_transformation',
        }
        if self.dispersion == 'feature':
            params['r'] = 'per_feature'
        else:
            raise NotImplementedError()
        return params

    def initial_transformation(self, x, x_mask=1.):
        x = x
        aux_info = {}
        aux_info[f'{self.aux_prefix}x_library_size'] = (x_mask * x).sum(dim=-1) if x_mask is not None else x.sum(dim=-1)
        x = torch.log1p(x / aux_info[f'{self.aux_prefix}x_library_size'].unsqueeze(-1) * 1e4)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        mean = parameters['mean']
        r = 1. + parameters['r']
        library_size = lib_y

        if self.mean_transformation == 'exp':
            trans_mean = torch.exp(mean)
            trans_mean = library_size.unsqueeze(-1) / 1e4 * trans_mean
        elif self.mean_transformation == 'softmax':
            trans_mean = torch.softmax(mean, dim=-1)
            trans_mean = library_size.unsqueeze(-1) * trans_mean
        else:
            raise NotImplementedError()
        trans_r = torch.exp(r)
        return NegativeBinomial(mu=trans_mean, theta=trans_r, scale=None)


class LogNegativeBinomial(Distribution):
    def __init__(self, log_m, log_r, eps: float = 1e-8, validate_args=False) -> None:
        self.log_m = log_m
        self.log_r = log_r
        self._eps = eps
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        return torch.exp(self.log_m)

    @property
    def theta(self):
        return torch.exp(self.log_r)

    @property
    def variance(self):
        return self.mean + (self.mean ** 2) / self.theta

    def sample(self, sample_shape):
        raise NotImplementedError()

    @staticmethod
    def negative_binomial_log_ver(k, m_log, r_log, eps=1e-8):
        # r :D
        r = torch.exp(r_log)

        # choice_part = log(binom(k, k+r-1))
        choice_part = torch.lgamma(k + r + eps) - torch.lgamma(k + 1 + eps) - torch.lgamma(r + eps)
        # log_pow_k = log(p ^ k) = log((m/(m+r)) ^ k)
        log_pow_k = - k * F.softplus(r_log - m_log + eps)
        # log_pow_r = log((1 - p) ^ r) = log((r/(m+r)) ^ r)
        log_pow_r = - r * F.softplus(m_log - r_log + eps)

        return choice_part + log_pow_k + log_pow_r

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.negative_binomial_log_ver(value, self.log_m, self.log_r, eps=self._eps)


class LogNegativeBinomialNoiseModel(NoiseModel):
    def __init__(self, dispersion='feature', aux_prefix='__nm_'):
        super().__init__(aux_prefix=aux_prefix)
        self.dispersion = dispersion

    @property
    def parameters(self):
        params = {
            'mean': 'no_transformation',
        }
        if self.dispersion == 'feature':
            params['r'] = 'per_feature'
        else:
            raise NotImplementedError()
        return params

    def initial_transformation(self, x, x_mask=1.):
        x = x
        aux_info = {}
        aux_info[f'{self.aux_prefix}x_library_size'] = (x_mask * x).sum(dim=-1) if x_mask is not None else x.sum(dim=-1)
        x = torch.log1p(x / aux_info[f'{self.aux_prefix}x_library_size'].unsqueeze(-1) * 1e4)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        mean = parameters['mean']
        r = 1. + parameters['r']
        library_size = lib_y

        trans_mean = mean + torch.log(library_size.unsqueeze(-1) / 1e4)
        trans_r = r
        return LogNegativeBinomial(log_m=trans_mean, log_r=trans_r)
