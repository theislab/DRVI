import merlin.dtypes
import merlin.schema
import pyarrow as pa


def read_first_row(path):
    """Read first row of a parquet file in a memory efficient way"""
    parquet_file = pa.parquet.ParquetFile(path)
    first_row = next(parquet_file.iter_batches(batch_size=1))
    df = pa.Table.from_batches([first_row])
    return df.to_pylist()[0]


def transfer_type_from_pyarrow(schema, first_row=None):
    """Transfer pyarrow schema to merlin schema."""
    if isinstance(schema, pa.Schema):
        fields = []
        for field in schema:
            coresponsing_first_row = None
            if first_row is not None:
                coresponsing_first_row = first_row[field.name]
            fields.append(transfer_type_from_pyarrow(field, first_row=coresponsing_first_row))
        return merlin.schema.Schema(fields)
    elif isinstance(schema, pa.Field):
        return merlin.schema.ColumnSchema(schema.name, **transfer_type_from_pyarrow(schema.type, first_row=first_row))
    elif isinstance(schema, pa.lib.ListType):
        return {
            "dtype": transfer_type_from_pyarrow(schema.value_type),
            "is_list": True,
            "is_ragged": False,
            "properties": {"value_count": {"max": len(first_row)}},
        }
    elif isinstance(schema, pa.DataType):
        if pa.float32().equals(schema):
            return {"dtype": merlin.dtypes.float32}
        elif pa.float64().equals(schema):
            return {"dtype": merlin.dtypes.float64}
        elif pa.int32().equals(schema):
            return {"dtype": merlin.dtypes.int32}
        elif pa.int64().equals(schema):
            return {"dtype": merlin.dtypes.int64}
        elif pa.bool_().equals(schema):
            return {"dtype": merlin.dtypes.boolean}
        else:
            raise ValueError(f"Unknown type for {schema}")
    else:
        raise ValueError(f"Unknown type for {schema}")
