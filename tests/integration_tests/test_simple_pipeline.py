import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import drvi


class TestSimplePipelineOfTrainingAndInterpretability:
    n = 50
    g = 20
    c = 2
    b = 5

    def make_test_adata(self, is_sparse=True):
        N, G, C, B = self.n, self.g, self.c, self.b

        ct_list = np.sort(np.random.choice(range(C), N))[:, np.newaxis]
        g_exp_list = np.sort(np.random.choice(range(C), G))[:, np.newaxis]
        ct_array = (np.indices((N, C))[1] == ct_list) + 0.0
        g_exp_array = (np.indices((G, C))[1] == g_exp_list) + 0.0

        batch_list = np.random.choice(range(B), N)[:, np.newaxis]
        batch_list_2 = np.random.choice(range(B), N)[:, np.newaxis]
        exp_indicator = ct_array @ g_exp_array.T
        g_mean_list = np.exp((np.random.random(G) + 0.5) * 2)[:, np.newaxis]

        exp_matrix = np.random.poisson(exp_indicator * g_mean_list.T).astype(np.float32)

        adata = ad.AnnData(
            X=sparse.csr_matrix(exp_matrix),
            obs=pd.DataFrame(
                {
                    "cell_type": [f"ct_{ct}" for ct in ct_list[:, 0]],
                    "batch": [f"batch_{bid}" for bid in batch_list[:, 0]],
                    "batch_2": [f"batch_{bid}" for bid in batch_list_2[:, 0]],
                },
                index=[f"cell_{i}" for i in range(N)],
            ),
            var=pd.DataFrame(
                {
                    "gene_mean": g_mean_list[:, 0],
                    "gene_active_signature": np.apply_along_axis(
                        lambda x: "".join(x), axis=1, arr=g_exp_array.astype(str)
                    ),
                },
                index=[f"gene_{i}" for i in range(G)],
            ),
        )

        adata.obs["total_counts"] = adata.X.sum(axis=1)
        adata.layers["counts"] = adata.X.copy()
        adata.layers["lognorm"] = np.log1p(adata.X)

        if not is_sparse:
            adata.X = adata.X.A
            for l in ["counts", "lognorm"]:
                adata.layers[l] = adata.layers[l].A

        return adata

    def test_whole_integration_and_interpretability(self):
        adata = self.make_test_adata()

        drvi.model.DRVI.setup_anndata(
            adata,
            categorical_covariate_keys=["batch"],
            layer="counts",
            is_count_data=True,
        )
        model = drvi.model.DRVI(
            adata,
            n_latent=8,
            encoder_dims=[128],
            decoder_dims=[128],
            gene_likelihood="pnb_softmax",
            categorical_covariates=["batch"],
            decoder_reuse_weights="everywhere",
        )
        print(model.module)
        model.train(accelerator="cpu", max_epochs=100)
        embed = ad.AnnData(model.get_latent_representation(), obs=adata.obs)

        drvi.utils.tl.set_latent_dimension_stats(model, embed)

        traverse_adata = drvi.utils.tl.traverse_latent(model, embed, n_samples=2, max_noise_std=0.0)
        drvi.utils.tl.calculate_differential_vars(traverse_adata)

        return {
            "adata": adata,
            "model": model,
            "embed": embed,
            "traverse_adata": traverse_adata,
        }

    def test_plotting_functions(self):
        train_results = self.test_whole_integration_and_interpretability()
        adata = train_results["adata"]
        embed = train_results["embed"]
        traverse_adata = train_results["traverse_adata"]

        # pre-processing
        sc.pp.neighbors(embed, use_rep="X", n_pcs=embed.X.shape[1])
        sc.tl.umap(embed)
        sc.pp.pca(embed)

        drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2)
        drvi.utils.pl.plot_latent_dims_in_umap(embed)

        drvi.utils.pl.plot_latent_dims_in_umap(embed)

        drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "cell_type", title_col="title")
        drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.0)

        dimensions_interpretability = drvi.utils.tools.iterate_on_top_differential_vars(
            traverse_adata, key="combined_score", score_threshold=0.0
        )

        sample_dim = dimensions_interpretability[0][0]
        drvi.utils.pl.show_differential_vars_scatter_plot(
            traverse_adata,
            key_x="max_possible",
            key_y="min_possible",
            key_combined="combined_score",
            dim_subset=[sample_dim],
            score_threshold=0.0,
        )
        drvi.utils.pl.plot_relevant_genes_on_umap(
            adata,
            embed,
            traverse_adata,
            traverse_adata_key="combined_score",
            dim_subset=[sample_dim],
            score_threshold=0.0,
        )
