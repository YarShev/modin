# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""
Dataframe exchange protocol implementation.

See more in https://data-apis.org/dataframe-protocol/latest/index.html.

Notes
-----
- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""

import collections
from typing import Optional, Iterable, Sequence

from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolDataframe,
)
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.utils import _inherit_docstrings
from .column import PandasProtocolColumn


@_inherit_docstrings(ProtocolDataframe)
class PandasProtocolDataframe(ProtocolDataframe):
    """
    A data frame class, with only the methods required by the interchange protocol defined.

    Instances of this (private) class are returned from ``modin.pandas.DataFrame.__dataframe__``
    as objects with the methods and attributes defined on this class.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string. Columns may be accessed by name or by position.
    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.

    Parameters
    ----------
    df : PandasDataframe
        A ``PandasDataframe`` object.
    nan_as_null : bool, default:False
        A keyword intended for the consumer to tell the producer
        to overwrite null values in the data with ``NaN`` (or ``NaT``).
        This currently has no effect; once support for nullable extension
        dtypes is added, this value should be propagated to columns.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.
    offset : int, default: 0
        The offset of the first element.
    """

    def __init__(
        self,
        df: PandasDataframe,
        nan_as_null: bool = False,
        allow_copy: bool = True,
        offset: int = 0,
    ) -> None:
        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy
        self._offset = offset

    # TODO: ``What should we return???``, remove before the changes are merged
    @property
    def metadata(self):
        # `index` isn't a regular column, and the protocol doesn't support row
        # labels - so we export it as pandas-specific metadata here.
        return {"pandas.index": self._df.index}

    def num_columns(self) -> int:
        return len(self._df.columns)

    def num_rows(self) -> int:
        return len(self._df.index)

    def num_chunks(self) -> int:
        return self._df._partitions.shape[0]

    def column_names(self) -> Iterable[str]:
        for col in self._df.columns:
            yield col

    def get_column(self, i: int) -> PandasProtocolColumn:
        return PandasProtocolColumn(
            self._df.mask(row_positions=None, col_positions=[i]),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def get_column_by_name(self, name: str) -> PandasProtocolColumn:
        return PandasProtocolColumn(
            self._df.mask(row_positions=None, col_labels=[name]),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def get_columns(self) -> Iterable[PandasProtocolColumn]:
        for name in self._df.columns:
            yield PandasProtocolColumn(
                self._df.mask(row_positions=None, col_labels=[name]),
                allow_copy=self._allow_copy,
                offset=self._offset,
            )

    def select_columns(self, indices: Sequence[int]) -> "PandasProtocolDataframe":
        if not isinstance(indices, collections.Sequence):
            raise ValueError("`indices` is not a sequence")

        return PandasProtocolDataframe(
            self._df.mask(row_positions=None, col_positions=indices),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def select_columns_by_name(self, names: Sequence[str]) -> "PandasProtocolDataframe":
        if not isinstance(names, collections.Sequence):
            raise ValueError("`names` is not a sequence")

        return PandasProtocolDataframe(
            self._df.mask(row_positions=None, col_labels=names),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable["PandasProtocolDataframe"]:
        offset = 0
        if n_chunks is None:
            for length in self._df._row_lengths:
                yield PandasProtocolDataframe(
                    self._df.mask(row_positions=range(length), col_positions=None),
                    allow_copy=self._allow_copy,
                    offset=offset,
                )
                offset += length
        else:
            new_row_lengths = self.num_rows() // n_chunks
            if self.num_rows() % n_chunks:
                # TODO: raise exception in this case?
                new_row_lengths += 1

            new_partitions = self._df._partition_mgr_cls.map_axis_partitions(
                0,
                self._df._partitions,
                lambda df: df,
                keep_partitioning=False,
                lengths=new_row_lengths,
            )
            new_df = self._df.__constructor__(
                new_partitions,
                self._df.index,
                self._df.columns,
                new_row_lengths,
                self._df._column_widths,
            )
            for length in new_df._row_lengths:
                yield PandasProtocolDataframe(
                    self._df.mask(row_positions=range(length), col_positions=None),
                    allow_copy=self._allow_copy,
                    offset=offset,
                )
                offset += length
