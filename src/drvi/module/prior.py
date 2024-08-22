import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

# Standard, VaMP, GMM from Karin's CSI repo


class Prior(nn.Module):
    def __init__(self):
        super().__init__()

    def kl(self, qz):
        raise NotImplementedError()


class StandardPrior(Prior):
    def __init__(self):
        super().__init__()

    def kl(self, qz):
        # 1 x N
        assert isinstance(qz, Normal)
        return kl_divergence(qz, Normal(torch.zeros_like(qz.mean), torch.ones_like(qz.mean)))


class VampPrior(Prior):
    # Adapted from https://github.com/jmtomczak/intro_dgm/main/vaes/vae_priors_example.ipynb
    # K - components, I - inputs, L - latent, N - samples
    def __init__(
        self,
        n_components,
        encoder,
        model_input,
        trainable_keys=("x",),
        fixed_keys=(),
        input_type="scvi",
        preparation_function=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.input_type = input_type
        self.preparation_function = preparation_function

        # pseudo inputs
        pi_aux_data = {}
        pi_tensor_data = {}
        for key in fixed_keys:
            if isinstance(model_input[key], torch.Tensor):
                pi_tensor_data[key] = torch.nn.Parameter(model_input[key], requires_grad=False)
            else:
                pi_aux_data[key] = model_input[key]
        for key in trainable_keys:
            pi_tensor_data[key] = torch.nn.Parameter(model_input[key], requires_grad=True)
            assert pi_tensor_data[key].shape[0] == n_components
        self.pi_aux_data = pi_aux_data
        self.pi_tensor_data = nn.ParameterDict(pi_tensor_data)

        # mixing weights
        self.w = torch.nn.Parameter(torch.zeros(n_components, 1, 1))  # K x 1 x 1

    def get_params(self):
        # u->encoder->mean, var
        original_mode = self.encoder.training
        self.encoder.train(False)
        if self.input_type == "scfemb":
            z = self.encoder({**self.pi_aux_data, **self.pi_tensor_data})
            output = z["qz_mean"], z["qz_var"]
        elif self.input_type == "scvi":
            x, args, kwargs = self.preparation_function({**self.pi_aux_data, **self.pi_tensor_data})
            q_m, q_v, latent = self.encoder(x, *args, **kwargs)
            output = q_m, q_v
        else:
            self.encoder.train(original_mode)
            raise NotImplementedError()
        self.encoder.train(original_mode)
        return output  # (K x L), (K x L)

    def log_prob(self, z):
        # Mixture of gaussian computed on K x N x L
        z = z.unsqueeze(0)  # 1 x N x L

        # u->encoder->mean, var
        m_p, v_p = self.get_params()  # (K x L), (K x L)
        m_p = m_p.unsqueeze(1)  # K x 1 x L
        v_p = v_p.unsqueeze(1)  # K x 1 x L

        # mixing probabilities
        w = torch.nn.functional.softmax(self.w, dim=0)  # K x 1 x 1

        # sum of log_p across components weighted by w
        log_prob = Normal(m_p, v_p.sqrt()).log_prob(z) + torch.log(w)  # K x N x L
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=False)  # N x L

        return log_prob  # N x L

    def kl(self, qz):
        assert isinstance(qz, Normal)
        z = qz.rsample()
        return qz.log_prob(z) - self.log_prob(z)

    def get_extra_state(self):
        return {
            "pi_aux_data": self.pi_aux_data,
            "input_type": self.input_type,
        }

    def set_extra_state(self, state):
        self.pi_aux_data = state["pi_aux_data"]
        self.input_type = state["input_type"]


class GaussianMixtureModelPrior(Prior):
    # Based on VampPrior class

    def __init__(
        self,
        n_components,
        n_latent,
        data=None,
        trainable_priors=True,
    ):
        # Do we need python 2 compatibility?
        super().__init__()

        if data is None:
            p_m = torch.rand(n_components, n_latent)  # K x L
            p_v = torch.ones(n_components, n_latent)  # K x L
        else:
            p_m = data[0]
            p_v = data[1]
        self.p_m = torch.nn.Parameter(p_m, requires_grad=trainable_priors)
        self.p_v = torch.nn.Parameter(p_v, requires_grad=trainable_priors)

        # mixing weights
        self.w = torch.nn.Parameter(torch.zeros(self.p_m.shape[0], 1, 1))  # K x 1 x 1

    def log_prob(self, z):
        # Mixture of gaussian computed on K x N x L
        z = z.unsqueeze(0)  # 1 x N x L

        m_p = self.p_m.unsqueeze(1)  # K x 1 x L
        v_p = self.p_v.unsqueeze(1)  # K x 1 x L

        # mixing probabilities
        w = torch.nn.functional.softmax(self.w, dim=0)  # K x 1 x 1

        # sum of log_p across components weighted by w
        log_prob = Normal(m_p, v_p.sqrt()).log_prob(z) + torch.log(w)  # K x N x L
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=False)  # N x L

        return log_prob  # N x L

    def kl(self, qz):
        assert isinstance(qz, Normal)
        z = qz.rsample()
        return qz.log_prob(z) - self.log_prob(z)
