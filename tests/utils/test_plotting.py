import pandas as pd
import pytest

import drvi


def test_plot_interpretability_scores_requires_dimension_numbers():
    scores = pd.DataFrame({"no number": [1.0, 0.5]}, index=["gene_0", "gene_1"])
    with pytest.raises(ValueError, match="No dimension number"):
        drvi.utils.pl.plot_interpretability_scores(scores, show=False)
