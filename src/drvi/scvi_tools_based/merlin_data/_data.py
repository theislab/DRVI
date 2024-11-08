import math
import os
from typing import Literal
from uuid import uuid4

import merlin.io
import pandas as pd
import pyarrow.parquet as pq
from scvi.data import _constants

from drvi.scvi_tools_based.merlin_data._utils import read_first_row, transfer_type_from_pyarrow


class MerlinData:
    """
    Wrapper for merlin data object.

    Parameters
    ----------
    data_path : str
        Path to merlin data.
    train_key : str, optional
        Key for train data. Default is 'train'.
    validation_key : str, optional
        Key for validation data. Default is 'val'.
    test_key : str, optional
        Key for test data. Default is None.
    default_track : str, optional
        Default track to use for data. Default is 'train'. Possible values are 'train', 'val', 'test'.
    layer_key : str, optional
        Key for layer data. Default is None, which will be replaced with 'X'.
    sub_sample_frac : float
        The fraction of data to subsample. Default is 1.0.
    var_names_col_num : int
        The column number of the variable names in the var.parquet file. Default is 0.
    """

    def __init__(
        self,
        data_path: str,
        train_key: str = "train",
        validation_key: str = "val",
        test_key: str | None = "test",
        default_track: Literal["train", "val", "test"] = "train",
        layer_key: str | None = None,
        sub_sample_frac: float = 1.0,
        var_names_col_num=0,
    ) -> None:
        self.data_path = data_path
        self.train_key = train_key
        self.validation_key = validation_key
        self.test_key = test_key
        self.default_track = self.set_default_track(default_track)
        self.layer_key = "X" if layer_key is None else layer_key
        self.sub_sample_frac = sub_sample_frac

        self.var_names_col_num = var_names_col_num

        self.uns = {}
        self.schema, self.first_row = self.resolve_schema()

        assert self.schema.get(self.layer_key) is not None

    def get_default_track(self):
        return self.default_track

    def set_default_track(self, track):
        assert track in ["train", "val", "test"]
        self.default_track = track
        return track

    def resolve_schema(self):
        """
        Resolve the schema of the merlin data.

        Returns
        -------
        schema : dict
            The resolved schema of the merlin data.
        first_row : dict
            The first row of the merlin data.
        """
        single_parquet_filename = os.path.join(self.data_path, self.train_key, "part.0.parquet")
        first_row = read_first_row(single_parquet_filename)
        parquet_schema = pq.read_schema(single_parquet_filename)
        merlin_schema = transfer_type_from_pyarrow(parquet_schema, first_row=first_row)
        return merlin_schema, first_row

    def has_col(self, key):
        """
        Check if the merlin data has a specific column.

        Parameters
        ----------
        key : str
            The column key to check.

        Returns
        -------
        bool
            True if the column exists, False otherwise.
        """
        return self.schema.get(key) is not None

    def get_categorical_mapping(self, key):
        """
        Get the categorical mapping for a specific column.

        Parameters
        ----------
        key : str
            The column key to get the mapping for.

        Returns
        -------
        dict
            The categorical mapping for the column.
        """
        lookup_table = pd.read_parquet(os.path.join(self.data_path, "categorical_lookup", f"{key}.parquet"))
        lookup_table.columns = ["col_value"]
        mapping = lookup_table.to_dict()["col_value"]
        # correct mapping (int key to str) for compatbility with anndata summarization (attrdict)
        mapping = {str(k): v for k, v in mapping.items()}
        return mapping

    @property
    def n_vars(self):
        """
        Get the number of variables in the merlin data.

        Returns
        -------
        int
            The number of variables.
        """
        return self.schema.get(self.layer_key).value_count.max

    @property
    def var(self):
        """
        Get the variable metadata in the merlin data.

        Returns
        -------
        pd.DataFrame
            The variable metadata.
        """
        return pd.read_parquet(os.path.join(self.data_path, "var.parquet"))

    @property
    def var_names(self):
        """
        Get the variable names in the merlin data.

        Returns
        -------
        List[str]
            The variable names.
        """
        return self.var.iloc[:, self.var_names_col_num]

    @property
    def is_view(self):
        """For compatibility with anndata"""
        return False

    def _set_uuid(self, overwrite: bool = False):
        """
        Set the UUID for the merlin data.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing UUID. Default is False.
        """
        if _constants._SCVI_UUID_KEY not in self.uns or overwrite:
            self.uns[_constants._SCVI_UUID_KEY] = str(uuid4())

    def __repr__(self) -> str:
        return f"MerlinData object with schema: {self.schema.to_pandas()}"

    @staticmethod
    def _get_data_files(base_path: str, ds_key_disk: str, sub_sample_frac: float = 1.0):
        """
        Get the data files for a specific key.

        Parameters
        ----------
        base_path : str
            The base path of the data.
        ds_key_disk : str
            The key on disk to get the data files for.
        sub_sample_frac : float, optional
            The fraction of data to subsample. Default is 1.0.

        Returns
        -------
        str or List[str]
            The path(s) to the data file(s).
        """
        if sub_sample_frac == 1.0:
            # if no subsampling -> just return base path and merlin takes care of the rest
            return os.path.join(base_path, ds_key_disk)
        else:
            files = [file for file in os.listdir(os.path.join(base_path, ds_key_disk)) if file.endswith(".parquet")]
            files = [
                os.path.join(base_path, ds_key_disk, file) for file in sorted(files, key=lambda x: int(x.split(".")[1]))
            ]
            return files[: math.ceil(sub_sample_frac * len(files))]

    def get_dataset(self, split, columns, **dataset_kwargs):
        """
        Get the dataset for a specific split and columns.

        Parameters
        ----------
        split : str
            The split to get the dataset for.
        columns : List[str]
            The columns to include in the dataset.
        dataset_kwargs : dict, optional
            Additional keyword arguments to pass to the dataset.

        Returns
        -------
        merlin.io.Dataset
            The dataset object.
        """
        if split == "default":
            split = self.default_track
        sub_sample_frac = self.sub_sample_frac
        part_size = {
            "train": "100MB",
            "val": "100MB",
            "test": "300MB",
        }[split]
        ds_key_disk = {
            "train": self.train_key,
            "val": self.validation_key,
            "test": self.test_key,
        }[split]
        return merlin.io.Dataset(
            self._get_data_files(self.data_path, ds_key_disk=ds_key_disk, sub_sample_frac=sub_sample_frac),
            engine="parquet",
            part_size=part_size,
            schema=self.schema.select_by_name(columns),
            **dataset_kwargs,
        )
