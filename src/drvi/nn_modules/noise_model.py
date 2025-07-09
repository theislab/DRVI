from typing import Any, Literal

import torch
from scvi.distributions import NegativeBinomial
from torch.distributions import Distribution, Normal, Poisson
from torch.nn import functional as F


class NoiseModel:
    """Abstract base class for noise models in variational autoencoders.

    This class defines the interface for different noise models that can be
    used in the decoder of variational autoencoders. Each noise model specifies
    how the output distribution should be parameterized and how the data should
    be transformed.

    Notes
    -----
    Subclasses must implement:
    - `parameters`: Property defining the required parameters
    - `dist`: Method creating the output distribution

    The noise model is responsible for:
    1. Defining what parameters the decoder should output
    2. Transforming the decoder outputs into distribution parameters
    3. Creating the appropriate distribution for the data
    """

    def __init__(self) -> None:
        pass

    @property
    def parameters(self) -> dict[str, str]:
        """Get the parameter specification for this noise model.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their specifications.
            Common specifications include:
            - "no_transformation": Parameter is used directly
            - "per_feature": Parameter is learned per feature
            - "fixed=value": Parameter is fixed to a specific value

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    @property
    def main_param(self) -> str:
        """Get the main parameter name for this noise model.

        Returns
        -------
        str
            Name of the main parameter (usually "mean").
        """
        return "mean"

    def initial_transformation(
        self, x: torch.Tensor, x_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply initial transformation to input data.

        Parameters
        ----------
        x
            Input data tensor.
        x_mask
            Mask for the input data.

        Returns
        -------
        tuple
            (transformed_x, aux_info) where aux_info contains additional
            information needed for the distribution.
        """
        x = x
        aux_info = {}
        return x, aux_info

    def dist(self, aux_info: dict[str, Any], parameters: dict[str, torch.Tensor], lib_y: torch.Tensor) -> Distribution:
        """Create the output distribution.

        Parameters
        ----------
        aux_info
            Additional information from initial_transformation.
        parameters
            Parameters from the decoder network.
        lib_y
            Library size tensor.

        Returns
        -------
        Distribution
            The output distribution.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        # lib_y is so strange here. I should think about it.
        raise NotImplementedError()


def calculate_library_size(x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Calculate the library size for each sample.

    Parameters
    ----------
    x
        Input data tensor.
    x_mask
        Mask for the input data.

    Returns
    -------
    torch.Tensor
        Library size for each sample.

    Notes
    -----
    The library size is the sum of all features for each sample,
    optionally weighted by the mask.
    """
    return (x_mask * x).sum(dim=-1) if x_mask is not None else x.sum(dim=-1)


def library_size_normalization(
    x: torch.Tensor,
    lib_size: torch.Tensor,
    library_normalization: Literal["none", "x_lib", "x_loglib", "div_lib_x_loglib", "x_loglib_all"],
) -> torch.Tensor:
    """Normalize data by library size.

    Parameters
    ----------
    x
        Input data tensor.
    lib_size
        Library size tensor.
    library_normalization
        Normalization method to apply.

    Returns
    -------
    torch.Tensor
        Normalized data tensor.

    Notes
    -----
    Different normalization methods:
    - "none": No normalization
    - "x_lib": Divide by library size and scale by 1e4
    - "x_loglib": No normalization (same as "none")
    - "div_lib_x_loglib": Divide by library size and scale by 1e4
    - "x_loglib_all": Divide by log of library size and scale by 1e1
    """
    # TODO: remove any library_normalization but 'none', 'x_lib'
    if library_normalization in ["none", "x_loglib"]:
        x = x
    elif library_normalization in ["x_lib", "div_lib_x_loglib"]:
        x = x / lib_size.unsqueeze(-1) * 1e4
    elif library_normalization in ["x_loglib_all"]:
        x = x / torch.log(lib_size.unsqueeze(-1)) * 1e1
    else:
        raise NotImplementedError()
    return x


def library_size_correction(
    x: torch.Tensor,
    lib_size: torch.Tensor,
    library_normalization: Literal["none", "x_lib", "x_loglib", "div_lib_x_loglib", "x_loglib_all"],
    log_space: bool = False,
) -> torch.Tensor:
    """Apply library size correction to data.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor.
    lib_size : torch.Tensor
        Library size tensor.
    library_normalization : {"none", "x_lib", "x_loglib", "div_lib_x_loglib", "x_loglib_all"}
        Normalization method that was applied.
    log_space : bool, default=False
        Whether the data is in log space.

    Returns
    -------
    torch.Tensor
        Corrected data tensor.

    Notes
    -----
    This function reverses the library size normalization to recover
    the original scale of the data. The correction depends on whether
    the data is in log space or not.
    """
    # TODO: remove any library_normalization but 'none', 'x_lib'
    if library_normalization in ["none"]:
        x = x
    elif library_normalization in ["x_lib"]:
        if not log_space:
            x = x * lib_size.unsqueeze(-1).clip(1) / 1e4
        else:
            x = x + torch.log(lib_size.unsqueeze(-1) / 1e4).clip(0)
    elif library_normalization in ["x_loglib", "div_lib_x_loglib", "x_loglib_all"]:
        if not log_space:
            x = x * torch.log(lib_size.unsqueeze(-1)) / 1e4
        else:
            x = x + torch.log(torch.log(lib_size.unsqueeze(-1)).clip(0) / 1e4).clip(0)
    else:
        raise NotImplementedError()
    return x


class NormalNoiseModel(NoiseModel):
    """Normal (Gaussian) noise model for continuous data.

    This noise model assumes the data follows a normal distribution.
    It can handle both fixed and learnable variance parameters.

    Parameters
    ----------
    model_var
        Variance modeling strategy:
        - "fixed": Use fixed variance of 1e-2
        - "fixed=value": Use fixed variance of specified value
        - "dynamic": Learn variance per sample
        - "feature": Learn variance per feature
    eps
        Small constant added to variance for numerical stability.
    """

    def __init__(self, model_var="fixed", eps=1e-8):
        super().__init__()
        self.model_var = model_var
        self.eps = eps

    @property
    def parameters(self):
        """Get parameter specification for normal noise model.

        Returns
        -------
        dict
            Parameter specification with mean and variance parameters.
        """
        if self.model_var == "fixed":
            var_desc = "fixed=1e-2"
        elif self.model_var.startswith("fixed="):
            var_desc = self.model_var
        elif self.model_var == "dynamic":
            var_desc = "no_transformation"
        elif self.model_var == "feature":
            var_desc = "per_feature"
        else:
            raise NotImplementedError()
        return {
            "mean": "no_transformation",
            "var": var_desc,
        }

    def initial_transformation(self, x, x_mask=1.0):
        """Apply initial transformation for normal noise model.

        Parameters
        ----------
        x
            Input data tensor.
        x_mask
            Mask for the input data.

        Returns
        -------
        tuple
            (transformed_x, aux_info) where aux_info is empty for normal model.
        """
        x = x
        aux_info = {}
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        """Create normal distribution.

        Parameters
        ----------
        aux_info
            Additional information (unused for normal model).
        parameters
            Dictionary containing 'mean' and 'var' parameters.
        lib_y
            Library size tensor (unused for normal model).

        Returns
        -------
        Normal
            Normal distribution with specified mean and variance.
        """
        var = parameters["var"]
        if self.model_var:
            var = torch.nan_to_num(torch.exp(var), posinf=100, neginf=0) + self.eps
        return Normal(parameters["mean"], torch.abs(var).sqrt())


class PoissonNoiseModel(NoiseModel):
    """Poisson noise model for count data.

    This noise model assumes the data follows a Poisson distribution,
    which is appropriate for count data like gene expression counts.

    Parameters
    ----------
    mean_transformation
        Transformation to apply to the mean parameter:
        - "exp": Exponential transformation
        - "softmax": Softmax transformation
    library_normalization
        Library size normalization method.

    Examples
    --------
    >>> import torch
    >>> # Create Poisson noise model
    >>> model = PoissonNoiseModel(mean_transformation="exp")
    >>> print(model.parameters)
    {'mean': 'no_transformation'}
    >>> # Test with sample data
    >>> x = torch.randint(0, 100, (10, 5))
    >>> transformed_x, aux_info = model.initial_transformation(x)
    >>> print(transformed_x.shape)  # torch.Size([10, 5])
    """

    def __init__(self, mean_transformation="exp", library_normalization="x_lib"):
        super().__init__()
        self.mean_transformation = mean_transformation
        self.library_normalization = library_normalization

    @property
    def parameters(self):
        """Get parameter specification for Poisson noise model.

        Returns
        -------
        dict
            Parameter specification with mean parameter.
        """
        return {
            "mean": "no_transformation",
        }

    def initial_transformation(self, x, x_mask=1.0):
        """Apply initial transformation for Poisson noise model.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        x_mask : torch.Tensor, default=1.0
            Mask for the input data.

        Returns
        -------
        tuple
            (transformed_x, aux_info) where aux_info contains library size.
        """
        x = x
        aux_info = {}
        aux_info["x_library_size"] = calculate_library_size(x, x_mask)
        x = library_size_normalization(x, aux_info["x_library_size"], self.library_normalization)
        x = torch.log1p(x)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        """Create Poisson distribution.

        Parameters
        ----------
        aux_info : dict
            Additional information from initial_transformation.
        parameters : dict
            Dictionary containing 'mean' parameter.
        lib_y : torch.Tensor
            Library size tensor.

        Returns
        -------
        Poisson
            Poisson distribution with transformed mean.
        """
        mean = parameters["mean"]
        library_size = lib_y

        if self.mean_transformation == "exp":
            trans_mean = torch.exp(mean)
            trans_mean = library_size_correction(trans_mean, library_size, self.library_normalization, log_space=False)
        elif self.mean_transformation == "softmax":
            trans_mean = torch.softmax(mean, dim=-1)
            trans_mean = library_size.unsqueeze(-1) * trans_mean
        else:
            raise NotImplementedError()
        return Poisson(trans_mean)


class NegativeBinomialNoiseModel(NoiseModel):
    """Negative binomial noise model for overdispersed count data.

    This noise model assumes the data follows a negative binomial distribution,
    which is appropriate for gene expression data that exhibits overdispersion.

    Parameters
    ----------
    dispersion : {"feature"}, default="feature"
        Dispersion parameter modeling strategy.
    mean_transformation : {"exp", "softmax", "softplus", "none"}, default="exp"
        Transformation to apply to the mean parameter.
    library_normalization : {"none", "x_lib", "x_loglib", "div_lib_x_loglib", "x_loglib_all"}, default="x_lib"
        Library size normalization method.
    """

    def __init__(self, dispersion="feature", mean_transformation="exp", library_normalization="x_lib"):
        super().__init__()
        assert mean_transformation in ["exp", "softmax", "softplus", "none"]
        self.dispersion = dispersion
        self.mean_transformation = mean_transformation
        self.library_normalization = library_normalization

    @property
    def parameters(self):
        """Get parameter specification for negative binomial noise model.

        Returns
        -------
        dict
            Parameter specification with mean and dispersion parameters.
        """
        params = {
            "mean": "no_transformation",
        }
        if self.dispersion == "feature":
            params["r"] = "per_feature"
        else:
            raise NotImplementedError()
        return params

    def initial_transformation(self, x, x_mask=1.0):
        """Apply initial transformation for negative binomial noise model.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        x_mask : torch.Tensor, default=1.0
            Mask for the input data.

        Returns
        -------
        tuple
            (transformed_x, aux_info) where aux_info contains library size.
        """
        x = x
        aux_info = {}
        aux_info["x_library_size"] = calculate_library_size(x, x_mask)
        x = library_size_normalization(x, aux_info["x_library_size"], self.library_normalization)
        x = torch.log1p(x)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        """Create negative binomial distribution.

        Parameters
        ----------
        aux_info : dict
            Additional information from initial_transformation.
        parameters : dict
            Dictionary containing 'mean' and 'r' parameters.
        lib_y : torch.Tensor
            Library size tensor.

        Returns
        -------
        NegativeBinomial
            Negative binomial distribution with transformed parameters.
        """
        mean = parameters["mean"]
        r = 1.0 + parameters["r"]
        library_size = lib_y

        if self.mean_transformation == "exp":
            trans_mean = torch.exp(mean)
            trans_mean = library_size_correction(trans_mean, library_size, self.library_normalization, log_space=False)
        elif self.mean_transformation == "softmax":
            trans_mean = torch.softmax(mean, dim=-1)
            trans_mean = library_size.unsqueeze(-1) * trans_mean
        # `softplus` and `none` for ablation. Useless in practice.
        elif self.mean_transformation == "softplus":
            trans_mean = F.softplus(mean)
            trans_mean = library_size_correction(trans_mean, library_size, self.library_normalization, log_space=False)
        elif self.mean_transformation == "none":
            trans_mean = mean
            trans_mean = library_size_correction(trans_mean, library_size, self.library_normalization, log_space=False)
        else:
            raise NotImplementedError()
        trans_r = torch.exp(r)
        return NegativeBinomial(mu=trans_mean, theta=trans_r, scale=None)


class LogNegativeBinomial(Distribution):
    """Log-space negative binomial distribution.

    This distribution represents a negative binomial distribution
    parameterized in log space for numerical stability.

    Parameters
    ----------
    log_m : torch.Tensor
        Log of the mean parameter.
    log_r : torch.Tensor
        Log of the dispersion parameter.
    eps : float, default=1e-8
        Small constant for numerical stability.
    validate_args : bool, default=False
        Whether to validate distribution arguments.

    Notes
    -----
    This distribution is parameterized in log space to avoid numerical
    issues with very small or large values. The actual mean and dispersion
    are computed as exp(log_m) and exp(log_r) respectively.
    """

    def __init__(self, log_m, log_r, eps: float = 1e-8, validate_args=False) -> None:
        self.log_m = log_m
        self.log_r = log_r
        self._eps = eps
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        """Get the mean of the distribution.

        Returns
        -------
        torch.Tensor
            Mean of the distribution.
        """
        return torch.exp(self.log_m)

    @property
    def theta(self):
        """Get the dispersion parameter.

        Returns
        -------
        torch.Tensor
            Dispersion parameter of the distribution.
        """
        return torch.exp(self.log_r)

    @property
    def variance(self):
        """Get the variance of the distribution.

        Returns
        -------
        torch.Tensor
            Variance of the distribution.
        """
        m = self.mean
        r = self.theta
        return m + m * m / r

    def sample(self, sample_shape):
        """Generate samples from the distribution.

        Parameters
        ----------
        sample_shape : torch.Size
            Shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            Samples from the distribution.
        """
        raise NotImplementedError("Sampling not implemented for LogNegativeBinomial")

    @staticmethod
    def negative_binomial_log_ver(k, m_log, r_log, eps=1e-8):
        """Compute log probability for negative binomial in log space.

        Parameters
        ----------
        k : torch.Tensor
            Observed values.
        m_log : torch.Tensor
            Log of mean parameter.
        r_log : torch.Tensor
            Log of dispersion parameter.
        eps : float, default=1e-8
            Small constant for numerical stability.

        Returns
        -------
        torch.Tensor
            Log probability of the observations.

        Notes
        -----
        This function computes the log probability of negative binomial
        observations when the parameters are given in log space. It uses
        the log-gamma function for numerical stability.
        """
        # r :D
        r = torch.exp(r_log)

        # choice_part = log(binom(k, k+r-1))
        choice_part = torch.lgamma(k + r + eps) - torch.lgamma(k + 1 + eps) - torch.lgamma(r + eps)
        # log_pow_k = log(p ^ k) = log((m/(m+r)) ^ k)
        log_pow_k = -k * F.softplus(r_log - m_log + eps)
        # log_pow_r = log((1 - p) ^ r) = log((r/(m+r)) ^ r)
        log_pow_r = -r * F.softplus(m_log - r_log + eps)

        return choice_part + log_pow_k + log_pow_r

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.negative_binomial_log_ver(value, self.log_m, self.log_r, eps=self._eps)


class LogNegativeBinomialNoiseModel(NoiseModel):
    """Log-space negative binomial noise model.

    This noise model uses a negative binomial distribution parameterized
    in log space for better numerical stability.

    Parameters
    ----------
    dispersion : {"feature"}, default="feature"
        Dispersion parameter modeling strategy.
    mean_transformation : {"none"}, default="none"
        Transformation to apply to the mean parameter.
    library_normalization : {"none", "x_lib", "x_loglib", "div_lib_x_loglib", "x_loglib_all"}, default="x_lib"
        Library size normalization method.

    Notes
    -----
    This model is similar to NegativeBinomialNoiseModel but uses log-space
    parameterization for better numerical stability, especially when dealing
    with very small or large values.
    """

    def __init__(self, dispersion="feature", mean_transformation="none", library_normalization="x_lib"):
        super().__init__()
        self.dispersion = dispersion
        self.mean_transformation = mean_transformation
        self.library_normalization = library_normalization

    @property
    def parameters(self):
        """Get parameter specification for log negative binomial noise model.

        Returns
        -------
        dict
            Parameter specification with mean and dispersion parameters.
        """
        params = {
            "mean": "no_transformation",
        }
        if self.dispersion == "feature":
            params["r"] = "per_feature"
        else:
            raise NotImplementedError()
        return params

    def initial_transformation(self, x, x_mask=1.0):
        """Apply initial transformation for log negative binomial noise model.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        x_mask : torch.Tensor, default=1.0
            Mask for the input data.

        Returns
        -------
        tuple
            (transformed_x, aux_info) where aux_info contains library size.
        """
        x = x
        aux_info = {}
        aux_info["x_library_size"] = calculate_library_size(x, x_mask)
        x = library_size_normalization(x, aux_info["x_library_size"], self.library_normalization)
        x = torch.log1p(x)
        return x, aux_info

    def dist(self, aux_info, parameters, lib_y):
        """Create log-space negative binomial distribution.

        Parameters
        ----------
        aux_info : dict
            Additional information from initial_transformation.
        parameters : dict
            Dictionary containing 'mean' and 'r' parameters.
        lib_y : torch.Tensor
            Library size tensor.

        Returns
        -------
        LogNegativeBinomial
            Log-space negative binomial distribution.
        """
        mean = parameters["mean"]
        r = 1.0 + parameters["r"]
        library_size = lib_y

        if self.mean_transformation == "none":
            trans_mean = library_size_correction(mean, library_size, self.library_normalization, log_space=True)
        elif self.mean_transformation == "softmax":
            trans_mean = mean - torch.logsumexp(mean, dim=-1, keepdim=True)
            trans_mean = torch.log(library_size).unsqueeze(-1) + trans_mean
        else:
            raise NotImplementedError()
        trans_r = r
        return LogNegativeBinomial(log_m=trans_mean, log_r=trans_r)
