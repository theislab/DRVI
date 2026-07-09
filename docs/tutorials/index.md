# Tutorials

Our tutorials demonstrate the practical application of DRVI across different use cases.

```{toctree}
:maxdepth: 1

external/general_pipeline
external/query_to_reference_mapping
external/find_rare_cell_types
external/identification_of_factors_1_cell_types
external/identification_of_factors_2_biological_processes
external/identification_of_factors_3_llm_tools
external/identification_of_factors_4_curation
external/porting_drvi_to_scvi_tools
```


## Installing tutorial dependencies

The tutorials need a few packages on top of the core DRVI stack. These are
provided as optional-dependency extras so you only install what a given tutorial
needs:

| Extra | Install | Covers |
| --- | --- | --- |
| `tutorials` | `pip install "drvi-py[tutorials]"` | Common packages for the general tutorials (e.g. general pipeline, rare cell types, query-to-reference mapping). |
| `tutorials-cell-types` | `pip install "drvi-py[tutorials-cell-types]"` | Identification of factors — cell types (adds `celltypist`, `networkx`). |
| `tutorials-biological-processes` | `pip install "drvi-py[tutorials-biological-processes]"` | Identification of factors — biological processes (adds `decoupler`, `gprofiler-official`, `gseapy`, `statsmodels`). |
| `tutorials-llm` | `pip install "drvi-py[tutorials-llm]"` | Identification of factors — LLM tools (adds `anthropic`, `cassia`, `claude-agent-sdk`, `google-genai`, `gs2txt`, `nest-asyncio`, `openai`). Most backends need an API key. |
| `tutorials-all` | `pip install "drvi-py[tutorials-all]"` | Everything for every tutorial, including all identification notebooks and the LLM backends. |

Extras can be combined, e.g. `pip install "drvi-py[tutorials,tutorials-cell-types]"`.

```{note}
The `tutorials-llm` backends are optional — you only need the one(s) you actually
use. Most require an API key, except `claude-agent-sdk`, which uses a logged-in
`claude` CLI. The "identification of factors — curation" tutorial reads results
produced by the earlier notebooks, so it needs no extra packages beyond core DRVI.
```
