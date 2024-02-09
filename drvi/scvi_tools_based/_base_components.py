import collections
import math
from typing import Callable, Iterable, Dict, Literal, Union, Sequence

import torch
from scvi.nn._utils import one_hot
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from drvi.module.embedding import MultiEmbedding
from drvi.module.freezable import FreezableBatchNorm1d, FreezableLayerNorm
from drvi.module.layer.factory import LayerFactory, FCLayerFactory
from drvi.module.layer.linear_layer import StackedLinearLayer
from drvi.module.noise_model import NoiseModel


def _identity(x):
    return x


class BatchWiseAdapter(nn.Module):
    def __init__(self, batch_repr_dim, input_dim, output_dim, low_rank_dim=10):
        super().__init__()
        self.L = nn.Parameter(torch.empty([batch_repr_dim, input_dim, low_rank_dim]))
        self.R = nn.Parameter(torch.empty([batch_repr_dim, low_rank_dim, output_dim]))
        self.L_bias = nn.Parameter(torch.empty([low_rank_dim]))
        self._reset_parameters()

    def _reset_parameters(self):
        gain = 1.
        fan_l = self.L.shape[0]
        std_l = gain / math.sqrt(fan_l)
        bound_l = math.sqrt(3.0) * std_l
        fan_r = self.R.shape[0]
        std_r = gain / math.sqrt(fan_r)
        bound_r = math.sqrt(3.0) * std_r

        with torch.no_grad():
            self.L.uniform_(-bound_l, bound_l)
            self.R.uniform_(-bound_r, bound_r)
            self.L_bias.zero_()

    def forward(self, x, batch_emb):
        # TODO: replace with more efficient ops
        l_t = torch.einsum('be,eid->bid', batch_emb, self.L)
        r_t = torch.einsum('be,edo->bdo', batch_emb, self.R)
        intermediate_tensor = F.relu(torch.einsum('b...i,bid->b...d', x, l_t) + self.L_bias)
        return torch.einsum('b...d,bdo->b...o', intermediate_tensor, r_t)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"batch_repr_dim={self.L.shape[0]}, "
                f"input_dim={self.L.shape[1]}, "
                f"output_dim={self.R.shape[2]}, "
                f"low_rank_dim={self.L.shape[2]}"
                f")")


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    layers_dim
        Number of nodes in layers including input and output dimensions
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    split_size
        The size of split if input is a 3d tensor otherwise -1
        This parameter is required to handle batch normalization
    reuse_weights
        Whether to reuse weights when having multiple splits
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    affine_batch_norm
        Whether to have affine transformation in `BatchNorm` layers
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    layer_factory
        A layer Factory instance to build projection layers based on
    layers_location
        An indicator to tell the class where in the architecture these layers reside.
    covariate_modeling_strategy
        The strategy model to consider covariates
    """

    def __init__(
            self,
            layers_dim: Sequence[int],
            n_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
            split_size: int = -1,
            reuse_weights: bool = True,
            use_batch_norm: bool = True,
            affine_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: nn.Module = nn.ReLU,
            layer_factory: LayerFactory = None,
            layers_location: Literal["intermediate", "first", "last"] = "intermediate",
            covariate_modeling_strategy: Literal[
                "one_hot", "emb", "emb_shared", "one_hot_linear", "emb_linear", "emb_shared_linear",
                "emb_adapter", "one_hot_adapter", "emb_shared_adapter"] = "one_hot",
            covariate_embs_dim: Iterable[int] = tuple([]),
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        if covariate_modeling_strategy.endswith("_linear"):
            self.covariate_projection_modeling = "linear"
            self.covariate_vector_modeling = covariate_modeling_strategy[:-len("_linear")]
        elif covariate_modeling_strategy.endswith("_adapter"):
            self.covariate_projection_modeling = "adapter"
            self.covariate_vector_modeling = covariate_modeling_strategy[:-len("_adapter")]
        else:
            self.covariate_projection_modeling = "cat"
            self.covariate_vector_modeling = covariate_modeling_strategy
        layer_factory = layer_factory or FCLayerFactory()

        self.n_cat_list = n_cat_list if n_cat_list is not None else []
        if self.covariate_vector_modeling == "one_hot":
            covariate_embs_dim = self.n_cat_list
        else:
            assert len(covariate_embs_dim) == len(self.n_cat_list)

        self.injectable_layers = []
        self.linear_batch_projections = []
        self.adapter_batch_projections = []

        def is_intermediate(i):
            assert layers_location in ["intermediate", "first", "last"]
            if layers_location == "first" and i == 0:
                return False
            if layers_location == "last" and i == len(layers_dim) - 2:
                return False
            return True
        
        def inject_into_layer(layer_num):
            user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
            return user_cond

        def get_projection_layer(n_in, n_out, i):
            output = []
            layer_needs_injection = False
            if not reuse_weights:
                assert split_size > 1
                if self.covariate_projection_modeling not in ["cat", "linear"]:
                    # cannot reuse adapters
                    raise NotImplementedError()

            if len(self.n_cat_list) > 0 and inject_into_layer(i):
                layer_needs_injection = True
                cat_dim = sum(covariate_embs_dim)
                if self.covariate_vector_modeling == "emb":
                    batch_emb = MultiEmbedding(
                        self.n_cat_list, covariate_embs_dim,
                        init_method='normal', max_norm=1.)
                    output.append(batch_emb)
                if self.covariate_projection_modeling == "cat":
                    n_in += cat_dim
                elif self.covariate_projection_modeling == "linear":
                    if reuse_weights:
                        linear_batch_projection = nn.Linear(cat_dim, n_out, bias=False)
                    else:
                        linear_batch_projection = StackedLinearLayer(split_size, cat_dim, n_out, bias=False)
                    output.append(linear_batch_projection)
                    self.linear_batch_projections.append(linear_batch_projection)
                elif self.covariate_projection_modeling == "adapter":
                    adapter_module = BatchWiseAdapter(cat_dim, n_in, n_out, low_rank_dim=10)
                    output.append(adapter_module)
                    self.adapter_batch_projections.append(adapter_module)
                else:
                    raise NotImplementedError()
            if reuse_weights:
                layer = layer_factory.get_normal_layer(
                    n_in,
                    n_out,
                    bias=bias,
                    intermediate_layer=is_intermediate(i),
                )
            else:
                layer = layer_factory.get_stacked_layer(
                    split_size,
                    n_in,
                    n_out,
                    bias=bias,
                    intermediate_layer=is_intermediate(i),
                )
            if layer_needs_injection:
                self.injectable_layers.append(layer)
            output.append(layer)
            return output

        def get_normalization_layers(n_out):
            output = []
            if split_size == -1:
                if use_batch_norm:
                    # non-default params come from defaults in original Tensorflow implementation
                    output.append(FreezableBatchNorm1d(n_out, momentum=0.01, eps=0.001,
                                                       affine=affine_batch_norm))
                if use_layer_norm:
                    output.append(FreezableLayerNorm(n_out, elementwise_affine=False))
            else:
                if use_batch_norm:
                    # non-default params come from defaults in original Tensorflow implementation
                    output.append(FreezableBatchNorm1d(n_out * split_size, momentum=0.01, eps=0.001,
                                                       affine=affine_batch_norm))
                if use_layer_norm:
                    output.append(FreezableLayerNorm(split_size, n_out, elementwise_affine=False))
            return output

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.ModuleList([p for p in [
                            *get_projection_layer(n_in, n_out, i),
                            *get_normalization_layers(n_out),
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ] if p is not None]),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def set_online_update_hooks(self, previous_n_cats_per_cov, n_cats_per_cov):
        """Set online update hooks."""
        if sum(previous_n_cats_per_cov) == sum(n_cats_per_cov):
            print("Nothing to make hook for!")
            return

        def make_hook_function(weight):
            w_size = weight.size()
            if weight.dim() == 2:
                # 2D tensors
                with torch.no_grad():
                    # Freeze gradients for normal nodes
                    if w_size[1] == sum(n_cats_per_cov):
                        transfer_mask = []
                    else:
                        transfer_mask = [torch.zeros([w_size[0], w_size[1] - sum(n_cats_per_cov)], device=weight.device)]
                    # Iterate over the categories and Freeze old caterogies and make new ones trainable
                    for n_cat_new, n_cat_old in zip(n_cats_per_cov, previous_n_cats_per_cov):
                        transfer_mask.append(torch.zeros([w_size[0], n_cat_old], device=weight.device))
                        if n_cat_new > n_cat_old:
                            transfer_mask.append(torch.ones([w_size[0], n_cat_new - n_cat_old], device=weight.device))
                    transfer_mask = torch.cat(transfer_mask, dim=1)
            elif weight.dim() == 3:                        
                # 3D tensors
                with torch.no_grad():
                    # Freeze gradients for normal nodes
                    if w_size[1] == sum(n_cats_per_cov):
                        transfer_mask = []
                    else:
                        transfer_mask = [torch.zeros([w_size[0], w_size[1] - sum(n_cats_per_cov), w_size[2]], device=weight.device)]
                    # Iterate over the categories and Freeze old caterogies and make new ones trainable
                    for n_cat_new, n_cat_old in zip(n_cats_per_cov, previous_n_cats_per_cov):
                        transfer_mask.append(torch.zeros([w_size[0], n_cat_old, w_size[2]], device=weight.device))
                        if n_cat_new > n_cat_old:
                            transfer_mask.append(torch.ones([w_size[0], n_cat_new - n_cat_old, w_size[2]], device=weight.device))
                    transfer_mask = torch.cat(transfer_mask, dim=1)
            else:
                raise NotImplementedError()

            def _hook_fn_injectable(grad):
                return grad * transfer_mask
            
            return _hook_fn_injectable

        if self.covariate_projection_modeling == "adapter":
            raise NotImplementedError()

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if self.covariate_vector_modeling == "emb_shared":
                    # Nothing to do here :)
                    pass
                elif self.covariate_vector_modeling == "emb":
                    # Freeze everything but embs (new embs)
                    if isinstance(layer, MultiEmbedding):
                        assert tuple(layer.num_embeddings) == tuple(n_cats_per_cov)
                        print(f"Freezing old categories for {layer}")
                        layer.freeze_top_embs(previous_n_cats_per_cov)
                    # No need to handle others. For them required_grad is already set to False
                elif self.covariate_vector_modeling == "one_hot":
                    assert self.covariate_projection_modeling in ["cat", "linear"]
                    # Freeze everything but linears right after one_hot (new weights)
                    if layer in self.injectable_layers or layer in self.linear_batch_projections:
                        layer.weight.requires_grad = True
                        print(f"Registering backward hook parameter with shape {layer.weight.size()}")
                        layer.weight.register_hook(make_hook_function(layer.weight))
                        if layer.bias is not None:
                            assert not layer.bias.requires_grad
                else:
                    raise NotImplementedError()

    def forward(self, x: torch.Tensor, cat_full_tensor: torch.Tensor):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_full_tensor
            Tensor of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        if self.covariate_vector_modeling == "one_hot":
            concat_list = []
            if cat_full_tensor is not None:
                cat_list = torch.split(cat_full_tensor, 1, dim=1)
            else:
                cat_list = tuple()

            if len(self.n_cat_list) > len(cat_list):
                raise ValueError(
                    "nb. categorical args provided doesn't match init. params."
                )
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat and cat is None:
                    raise ValueError("cat not provided while n_cat != 0 in init. params.")
                concat_list += [one_hot(cat, n_cat)]
        elif self.covariate_vector_modeling == "emb_shared":
            concat_list = [cat_full_tensor]
        else:
            concat_list = []

        def dimension_transformation(t):
            if x.dim() == t.dim():
                return t
            if x.dim() == 3 and t.dim() == 2:
                return t.unsqueeze(dim=1).expand(-1, x.shape[1], -1)
            raise NotImplementedError()

        for i, layers in enumerate(self.fc_layers):
            concat_list_layer = concat_list
            projected_batch_layer = None
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = layer(x.reshape(x.shape[0], -1)).reshape(x.shape)
                        else:
                            x = layer(x)
                    elif isinstance(layer, MultiEmbedding):
                        assert self.covariate_vector_modeling in ["emb"]
                        assert len(concat_list) == 0
                        concat_list_layer = [layer(cat_full_tensor.int())]
                    elif layer in self.linear_batch_projections:
                        assert self.covariate_projection_modeling == "linear"
                        projected_batch_layer = layer(torch.cat(concat_list_layer, dim=-1))
                    elif layer in self.adapter_batch_projections:
                        assert self.covariate_projection_modeling == "adapter"
                        projected_batch_layer = layer(x, torch.cat(concat_list_layer, dim=-1))
                    else:
                        if layer in self.injectable_layers:
                            if self.covariate_projection_modeling == "cat":
                                current_cat_tensor = dimension_transformation(torch.cat(concat_list_layer, dim=-1))
                                x = torch.cat((x, current_cat_tensor), dim=-1)
                                x = layer(x)
                            elif self.covariate_projection_modeling in ["linear", "adapter"]:
                                x = layer(x) + dimension_transformation(projected_batch_layer)
                            else:
                                raise NotImplementedError()
                        else:
                            x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    layers_dim
        The number of nodes per hidden layer as a sequence
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    affine_batch_norm
        Whether to use affine in batch norms
    use_layer_norm
        Whether to use layer norm in layers
    input_dropout_rate
        Dropout rate to apply to the input
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    layer_factory
        A layer Factory instance for building layers
    layers_location
        A hint on where the layer resides in the model
    covariate_modeling_strategy
        The strategy model takes to model covariates
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            layers_dim: Sequence[int] = tuple([128]),
            n_cat_list: Iterable[int] = None,
            inject_covariates: bool = True,
            use_batch_norm: bool = True,
            affine_batch_norm: bool = True,
            use_layer_norm: bool = False,
            input_dropout_rate: float = 0.,
            dropout_rate: float = 0.1,
            distribution: str = "normal",
            var_eps: float = 1e-4,
            var_activation: Union[Callable, Literal["exp", "pow2"]] = 'exp',
            layer_factory: LayerFactory = None,
            covariate_modeling_strategy: Literal[
                "one_hot", "emb", "emb_shared", "one_hot_linear", "emb_linear", "emb_shared_linear",
                "emb_adapter", "one_hot_adapter", "emb_shared_adapter"] = "one_hot",
            categorical_covariate_dims: Sequence[int] = tuple([]),
            return_dist: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.input_dropout = nn.Dropout(p=input_dropout_rate)

        all_layers_dim = [n_input] + list(layers_dim) + [n_output]
        if len(layers_dim) >= 1:
            self.encoder = FCLayers(
                layers_dim=all_layers_dim[:-1],
                n_cat_list=n_cat_list,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                affine_batch_norm=affine_batch_norm,
                use_layer_norm=use_layer_norm,
                inject_covariates=inject_covariates,
                layer_factory=layer_factory,
                layers_location="first",
                covariate_modeling_strategy=covariate_modeling_strategy,
                covariate_embs_dim=categorical_covariate_dims,
                **kwargs,
            )
        else:
            self.register_parameter('encoder', None)
            inject_covariates = True
        self.mean_encoder = FCLayers(
            layers_dim=all_layers_dim[-2:],
            n_cat_list=n_cat_list if inject_covariates else [],
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=False,
            bias=True,
            dropout_rate=0,
            layer_factory=layer_factory,
            layers_location="intermediate" if len(layers_dim) >= 1 else "first",
            covariate_modeling_strategy=covariate_modeling_strategy,
            covariate_embs_dim=categorical_covariate_dims if inject_covariates else [],
            **kwargs,
        )
        self.var_encoder = FCLayers(
            layers_dim=all_layers_dim[-2:],
            n_cat_list=n_cat_list if inject_covariates else [],
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=False,
            bias=True,
            dropout_rate=0,
            layer_factory=layer_factory,
            layers_location="intermediate" if len(layers_dim) >= 1 else "first",
            covariate_modeling_strategy=covariate_modeling_strategy,
            covariate_embs_dim=categorical_covariate_dims if inject_covariates else [],
            **kwargs,
        )
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        if var_activation == 'exp':
            self.var_activation = torch.exp
        elif var_activation == 'pow2':
            self.var_activation = lambda x: torch.pow(x, 2)
        else:
            assert callable(var_activation)
            self.var_activation = var_activation

    def forward(self, x: torch.Tensor, cat_full_tensor: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_full_tensor
            Tensor containing encoding of categorical variables of size n_batch x n_total_cat

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(self.input_dropout(x), cat_full_tensor) if self.encoder is not None else x
        q_m = self.mean_encoder(q, cat_full_tensor)
        q_v = self.var_activation(self.var_encoder(q, cat_full_tensor)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


# Decoder
class DecoderDRVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    layers_dim
        The number of nodes per hidden layer as a sequence
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_split
        The number of splits for latent dim
    split_aggregation
        How to aggregate splits in the last layer of the decoder
    split_method
        How to make splits. Can be 'split' or 'power', or 'split_map'.
    reuse_weights
        Were to reuse the weights of the decoder layers when using splitting
        Possible values are 'everywhere', 'last', 'intermediate', 'nowhere'.
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    affine_batch_norm
        Whether to use affine in batch norms
    use_layer_norm
        Whether to use layer norm in layers
    layer_factory
        A layer Factory instance for building layers
    covariate_modeling_strategy
        The strategy model takes to model covariates
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            gene_likelihood_module: NoiseModel,
            n_cat_list: Iterable[int] = None,
            n_split: int = 1,
            split_aggregation: Literal["sum", "logsumexp", "max"] = "logsumexp",
            split_method: Literal["split", "power", "split_map"] = "split",
            reuse_weights: Literal["everywhere", "last", "intermediate", "nowhere"] = "everywhere",
            layers_dim: Sequence[int] = tuple([128]),
            dropout_rate: float = 0.1,
            inject_covariates: bool = True,
            use_batch_norm: bool = False,
            affine_batch_norm: bool = True,
            use_layer_norm: bool = False,
            layer_factory: LayerFactory = None,
            covariate_modeling_strategy: Literal[
                "one_hot", "emb", "emb_shared", "one_hot_linear", "emb_linear", "emb_shared_linear",
                "emb_adapter", "one_hot_adapter", "emb_shared_adapter"] = "one_hot",
            categorical_covariate_dims: Sequence[int] = tuple([]),
            **kwargs,
    ):
        super().__init__()
        self.n_output = n_output
        self.gene_likelihood_module = gene_likelihood_module

        self.split_method = split_method
        self.n_split = n_split

        if n_split == -1 or n_split == 1:
            assert reuse_weights
            effective_dim = n_input
            self.n_split = n_split = -1
        elif self.split_method == "split":
            assert n_input % n_split == 0
            effective_dim = n_input // n_split
        elif self.split_method == "split_map":
            assert n_input % n_split == 0
            effective_dim = n_input
            self.split_transformation_weight = nn.Parameter(torch.randn(n_split, n_input // n_split, n_input))
            self.split_transformation_weight.data /= float(n_input // n_split) ** 0.5
        elif self.split_method == "power":
            effective_dim = n_input
            self.split_transformation = nn.Sequential(nn.Linear(n_input, n_input * n_split), nn.ReLU())
        else:
            raise NotImplementedError()

        self.effect_dim = effective_dim
        self.split_aggregation = split_aggregation

        assert reuse_weights in ["everywhere", "last", "intermediate", "nowhere"]
        intermediate_layers_reuse_weights = reuse_weights in ["everywhere", "intermediate"]
        last_layers_reuse_weights = reuse_weights in ["everywhere", "last"]

        all_layers_dim = [effective_dim] + list(layers_dim) + [n_output]
        if len(layers_dim) >= 1:
            self.px_shared_decoder = FCLayers(
                layers_dim=all_layers_dim[:-1],
                split_size=n_split,
                reuse_weights=intermediate_layers_reuse_weights,
                n_cat_list=n_cat_list,
                dropout_rate=dropout_rate,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                affine_batch_norm=affine_batch_norm,
                use_layer_norm=use_layer_norm,
                layer_factory=layer_factory,
                layers_location="intermediate",
                covariate_modeling_strategy=covariate_modeling_strategy,
                covariate_embs_dim=categorical_covariate_dims,
                **kwargs,
            )
        else:
            self.register_parameter('px_shared_decoder', None)
            inject_covariates = True

        params_for_likelihood = self.gene_likelihood_module.parameters
        params_nets = {}
        for param_name, param_info in params_for_likelihood.items():
            if param_info.startswith("fixed="):
                params_nets[param_name] = torch.nn.Parameter(
                    torch.tensor(float(param_info.split("=")[1])), requires_grad=False)
            elif param_info == 'no_transformation':
                params_nets[param_name] = FCLayers(
                    layers_dim=all_layers_dim[-2:],
                    split_size=n_split,
                    reuse_weights=last_layers_reuse_weights,
                    n_cat_list=n_cat_list if inject_covariates else [],
                    use_activation=False,
                    use_batch_norm=False,
                    use_layer_norm=False,
                    bias=True,
                    dropout_rate=0,
                    layer_factory=layer_factory,
                    layers_location="last",
                    covariate_modeling_strategy=covariate_modeling_strategy,
                    covariate_embs_dim=categorical_covariate_dims if inject_covariates else [],
                    **kwargs,
                )
            elif param_info == 'per_feature':
                params_nets[param_name] = torch.nn.Parameter(torch.randn(n_output))
            else:
                raise NotImplementedError()
        self.params_nets = nn.ParameterDict(params_nets)

    def forward(
            self,
            z: torch.Tensor,
            cat_full_tensor: torch.Tensor,
            library: torch.Tensor,
            gene_likelihood_additional_info: Dict,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        cat_full_tensor
            Tensor containing encoding of categorical variables of size n_batch x n_total_cat
        library
            library size
        gene_likelihood_additional_info
            additional info returned by gene likelihood module

        Returns
        -------
        distribution
        """
        batch_size = z.shape[0]
        if self.n_split > 1:
            if self.split_method == "power":
                z = self.split_transformation(z)
            z = torch.reshape(z, (batch_size, self.n_split, -1))
            if self.split_method == "split_map":
                z = torch.einsum('bsd,sdn->bsn', z, self.split_transformation_weight)

        last_tensor = self.px_shared_decoder(z, cat_full_tensor) if self.px_shared_decoder is not None else z
        original_params = {}
        params = {}
        for param_name, param_info in self.gene_likelihood_module.parameters.items():
            param_net = self.params_nets[param_name]
            if param_info.startswith("fixed="):
                original_params[param_name] = param_net
                params[param_name] = param_net.reshape(1, 1).expand(batch_size, self.n_output)
            elif param_info == 'no_transformation':
                param_value = param_net(last_tensor, cat_full_tensor)
                original_params[param_name] = param_value
                if self.n_split > 1:
                    if self.split_aggregation == "sum":
                        # to get average
                        params[param_name] = param_value.sum(dim=-2) / self.n_split
                    elif self.split_aggregation == "logsumexp":
                        # to cancel the effect of n_splits
                        params[param_name] = torch.logsumexp(param_value, dim=-2) - math.log(self.n_split)
                    elif self.split_aggregation == "max":
                        params[param_name] = torch.amax(param_value, dim=-2)
                    else:
                        raise NotImplementedError()
                else:
                    params[param_name] = param_value
            elif param_info == 'per_feature':
                original_params[param_name] = param_net
                params[param_name] = param_net.unsqueeze(0).expand(batch_size, -1)
            else:
                raise NotImplementedError()

        # Note this logic:
        px_dist = self.gene_likelihood_module.dist(aux_info=gene_likelihood_additional_info,
                                                   parameters=params, lib_y=library)
        return px_dist, params, original_params
