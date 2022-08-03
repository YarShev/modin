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

"""Module houses class that implements ``PandasDataframe`` using Ray."""

import pandas
from pandas.core.indexes.api import ensure_index
import numpy as np
import ray

from ..partitioning.partition_manager import PandasOnRayDataframePartitionManager
from modin.core.dataframe.pandas.dataframe.dataframe import (
    PandasDataframe,
    lazy_metadata_decorator,
)
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.error_message import ErrorMessage


@ray.remote
def _compute_dtype(dtypes, columns):
    return pandas.Series([np.dtype(dtypes)] * len(columns), index=columns)


class PandasOnRayDataframe(PandasDataframe):
    """
    The class implements the interface in ``PandasDataframe`` using Ray.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a ``pandas.Index``.
    columns : sequence
        The columns object for the dataframe. Converted to a ``pandas.Index``.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    """

    _partition_mgr_cls = PandasOnRayDataframePartitionManager

    def __init__(
        self,
        partitions,
        index=None,
        columns=None,
        row_lengths=None,
        column_widths=None,
        dtypes=None,
    ):
        self._index_cache = (
            ensure_index(index)
            if index is not None and not isinstance(index, ObjectIDType)
            else index
        )
        self._columns_cache = (
            ensure_index(columns)
            if columns is not None and not isinstance(columns, ObjectIDType)
            else columns
        )
        super().__init__(
            partitions,
            # index=index_cache,
            # columns=columns_cache,
            row_lengths=row_lengths,
            column_widths=column_widths,
            dtypes=dtypes,
        )

    def _validate_axes_lengths(self):
        """Validate that labels are split correctly if split is known."""
        if (
            self._row_lengths_cache is not None
            and not isinstance(self._row_lengths_cache, ObjectIDType)
            and len(self.index) > 0
        ):
            # An empty frame can have 0 rows but a nonempty index. If the frame
            # does have rows, the number of rows must equal the size of the
            # index.
            num_rows = sum(self._row_lengths_cache)
            if num_rows > 0:
                ErrorMessage.catch_bugs_and_request_email(
                    num_rows != len(self._index_cache),
                    f"Row lengths: {num_rows} != {len(self._index_cache)}",
                )
            ErrorMessage.catch_bugs_and_request_email(
                any(val < 0 for val in self._row_lengths_cache),
                f"Row lengths cannot be negative: {self._row_lengths_cache}",
            )
        if (
            self._column_widths_cache is not None
            and not isinstance(self._column_widths_cache, ObjectIDType)
            and len(self.columns) > 0
        ):
            # An empty frame can have 0 column but a nonempty column index. If
            # the frame does have columns, the number of columns must equal the
            # size of the columns.
            num_columns = sum(self._column_widths_cache)
            if num_columns > 0:
                ErrorMessage.catch_bugs_and_request_email(
                    num_columns != len(self._columns_cache),
                    f"Column widths: {num_columns} != {len(self._columns_cache)}",
                )
            ErrorMessage.catch_bugs_and_request_email(
                any(val < 0 for val in self._column_widths_cache),
                f"Column widths cannot be negative: {self._column_widths_cache}",
            )

    def _filter_empties(self, compute_metadata=True):
        """
        Remove empty partitions from `self._partitions` to avoid triggering excess computation.

        Parameters
        ----------
        compute_metadata : bool, default: True
            Trigger the computations for partition sizes and labels if they're not done already.
        """
        if not compute_metadata and (
            self._index_cache is None
            or self._columns_cache is None
            or self._row_lengths_cache is None
            or self._column_widths_cache is None
        ):
            # do not trigger the computations
            return

        if len(self.axes[0]) == 0 or len(self.axes[1]) == 0:
            # This is the case for an empty frame. We don't want to completely remove
            # all metadata and partitions so for the moment, we won't prune if the frame
            # is empty.
            # TODO: Handle empty dataframes better
            return
        self._partitions = np.array(
            [
                [
                    self._partitions[i][j]
                    for j in range(len(self._partitions[i]))
                    if j < len(self._column_widths) and self._column_widths[j] != 0
                ]
                for i in range(len(self._partitions))
                if i < len(self._row_lengths) and self._row_lengths[i] != 0
            ]
        )
        self._column_widths_cache = [w for w in self._column_widths if w != 0]
        self._row_lengths_cache = [r for r in self._row_lengths if r != 0]

    @property
    def _row_lengths(self):
        """
        Compute the row partitions lengths if they are not cached.

        Returns
        -------
        list
            A list of row partitions lengths.
        """
        if isinstance(self._row_lengths_cache, ObjectIDType):
            if len(self._partitions) > 0:
                self._row_lengths_cache = ray.get(self._row_lengths_cache)
            else:
                self._row_lengths_cache = []
            return self._row_lengths_cache
        else:
            return super()._row_lengths

    @property
    def _column_widths(self):
        """
        Compute the column partitions widths if they are not cached.

        Returns
        -------
        list
            A list of column partitions widths.
        """
        if isinstance(self._column_widths_cache, ObjectIDType):
            if len(self._partitions) > 0:
                self._column_widths_cache = ray.get(self._column_widths_cache)
            else:
                self._column_widths_cache = []
            return self._column_widths_cache
        else:
            return super()._column_widths

    def _compute_axis_labels_and_lengths_async(self, axis: int, partitions=None):
        """
        Compute the labels for specific `axis`.

        Parameters
        ----------
        axis : int
            Axis to compute labels along.
        partitions : np.ndarray, optional
            A 2D NumPy array of partitions from which labels will be grabbed.
            If not specified, partitions will be taken from `self._partitions`.

        Returns
        -------
        pandas.Index
            Labels for the specified `axis`.
        List of int
            Size of partitions alongside specified `axis`.
        """
        if partitions is None:
            partitions = self._partitions
        new_index, internal_idx = self._partition_mgr_cls.get_indices_async(
            axis, partitions
        )
        return new_index, internal_idx

    def _make_init_labels_args(self, partitions, index, columns) -> dict:
        kw = {}
        if index is None:
            kw["index"], kw["row_lengths"] = self._compute_axis_labels_and_lengths_async(0, partitions)
        else:
            kw["index"] = index
            _, kw["row_lengths"] = self._compute_axis_labels_and_lengths_async(0, partitions)
        
        if columns is None:
            kw["columns"], kw["column_widths"] = self._compute_axis_labels_and_lengths_async(1, partitions)
        else:
            kw["columns"] = columns
            _, kw["column_widths"] = self._compute_axis_labels_and_lengths_async(1, partitions)

        return kw

    def _get_index(self):
        """
        Get the index from the cache object.

        Returns
        -------
        pandas.Index
            An index object containing the row labels.
        """
        if isinstance(self._index_cache, ObjectIDType):
            self._index_cache = ray.get(self._index_cache)
            return self._index_cache
        else:
            return super()._get_index()

    def _get_columns(self):
        """
        Get the columns from the cache object.

        Returns
        -------
        pandas.Index
            An index object containing the column labels.
        """
        if isinstance(self._columns_cache, ObjectIDType):
            self._columns_cache = ray.get(self._columns_cache)
            return self._columns_cache
        else:
            return super()._get_columns()

    def _set_index(self, new_index):
        """
        Replace the current row labels with new labels.

        Parameters
        ----------
        new_index : list-like
            The new row labels.
        """
        if self._index_cache is None:
            self._index_cache = ensure_index(new_index)
        else:
            if isinstance(self._index_cache, ObjectIDType):
                self._index_cache = ray.get(self._index_cache)
            new_index = self._validate_set_axis(new_index, self._index_cache)
            self._index_cache = new_index
        self.synchronize_labels(axis=0)

    def _set_columns(self, new_columns):
        """
        Replace the current column labels with new labels.

        Parameters
        ----------
        new_columns : list-like
           The new column labels.
        """
        if self._columns_cache is None:
            self._columns_cache = ensure_index(new_columns)
        else:
            if isinstance(self._column_widths, ObjectIDType):
                self._columns_cache = ray.get(self._columns_cache)
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns
            if self._dtypes is not None and not isinstance(self._dtypes, ObjectIDType):
                self._dtypes.index = new_columns
        self.synchronize_labels(axis=1)

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    @property
    def dtypes(self):
        """
        Compute the data types if they are not cached.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """
        if isinstance(self._dtypes, ObjectIDType):
            self._dtypes = ray.get(self._dtypes)
            return self._dtypes
        return super().dtypes

    def _compute_dtypes_async(self, dtypes, columns):
        """
        Compute the data types via TreeReduce pattern.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """
        return _compute_dtype.remote(dtypes, columns)

    @lazy_metadata_decorator(apply_axis="both")
    def broadcast_apply_full_axis(
        self,
        axis,
        func,
        other,
        new_index=None,
        new_columns=None,
        apply_indices=None,
        enumerate_partitions=False,
        dtypes=None,
        check_axes_sync=True,
    ):
        """
        Broadcast partitions of `other` Modin DataFrame and apply a function along full axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply over (0 - rows, 1 - columns).
        func : callable
            Function to apply.
        other : PandasDataframe or list
            Modin DataFrame(s) to broadcast.
        new_index : list-like, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            Columns of the result. We may know this in
            advance, and if not provided it must be computed.
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether pass partition index into applied `func` or not.
            Note that `func` must be able to obtain `partition_idx` kwarg.
        dtypes : list-like, default: None
            Data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.
        check_axes_sync : bool, default: True
            In some cases, synchronization of the external index with the internal
            indexes of partitions is not required, since when performing certain
            functions, we know in advance that it must match. A good example is the
            `reindex` function. Although synchronization is performed asynchronously,
            it is nevertheless a rather expensive operation.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        if other is not None:
            if not isinstance(other, list):
                other = [other]
            other = [o._partitions for o in other] if len(other) else None

        if apply_indices is not None:
            numeric_indices = self.axes[axis ^ 1].get_indexer_for(apply_indices)
            apply_indices = self._get_dict_of_block_index(
                axis ^ 1, numeric_indices
            ).keys()

        new_partitions = self._partition_mgr_cls.broadcast_axis_partitions(
            axis=axis,
            left=self._partitions,
            right=other,
            apply_func=self._build_treereduce_func(axis, func),
            apply_indices=apply_indices,
            enumerate_partitions=enumerate_partitions,
            keep_partitioning=True,
        )
        # Index objects for new object creation. This is shorter than if..else
        kw = self._make_init_labels_args(new_partitions, new_index, new_columns)
        if dtypes == "copy":
            kw["dtypes"] = self._dtypes
        elif dtypes is not None:
            kw["dtypes"] = self._compute_dtypes_async(dtypes, kw["columns"])
        result = self.__constructor__(new_partitions, **kw)
        # There is can be case where synchronization is not needed
        if check_axes_sync and new_index is not None:
            result.synchronize_labels(axis=0)
        if check_axes_sync and new_columns is not None:
            result.synchronize_labels(axis=1)
        return result
