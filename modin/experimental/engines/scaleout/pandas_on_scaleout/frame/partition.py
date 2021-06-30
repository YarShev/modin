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

import pandas

from modin.engines.base.frame.partition import PandasFramePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas

import scaleout


class PandasOnScaleoutFramePartition(PandasFramePartition):
    def __init__(self, object_id, length=None, width=None, ip=None, call_queue=None):
        assert scaleout.is_object_ref(object_id)

        self.oid = object_id
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self._ip_cache = ip

    def get(self):
        """Gets the object out of the plasma store.

        Returns:
            The object from the plasma store.
        """
        if len(self.call_queue):
            self.drain_call_queue()
        return scaleout.get(self.oid)

    def apply(self, func, *args, **kwargs):
        """Apply a function to the object stored in this partition.

        Note: It does not matter if func is callable or an ObjectRef. Ray will
            handle it correctly either way. The keyword arguments are sent as a
            dictionary.

        Args:
            func: The function to apply.

        Returns:
            A RayRemotePartition object.
        """
        oid = self.oid
        call_queue = self.call_queue + [[func, args, kwargs]]
        if len(call_queue) > 1:
            result, length, width, ip = apply_list_of_funcs.remote(call_queue, oid)
        else:
            func, args, kwargs = call_queue[0]
            result, length, width, ip = apply_func.remote(oid, func, *args, **kwargs)
        return PandasOnScaleoutFramePartition(result, length, width, ip)

    def add_to_apply_calls(self, func, *args, **kwargs):
        return PandasOnScaleoutFramePartition(
            self.oid, call_queue=self.call_queue + [[func, args, kwargs]]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        oid = self.oid
        call_queue = self.call_queue
        if len(call_queue) > 1:
            (
                self.oid,
                self._length_cache,
                self._width_cache,
                self._ip_cache,
            ) = apply_list_of_funcs.remote(call_queue, oid)
        else:
            func, args, kwargs = call_queue[0]
            (
                self.oid,
                self._length_cache,
                self._width_cache,
                self._ip_cache,
            ) = apply_func.remote(oid, func, *args, **kwargs)
        self.call_queue = []

    def wait(self):
        self.drain_call_queue()
        scaleout.wait([self.oid])

    def __copy__(self):
        return PandasOnScaleoutFramePartition(
            self.oid,
            length=self._length_cache,
            width=self._width_cache,
            ip=self._ip_cache,
            call_queue=self.call_queue,
        )

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series
        return dataframe

    def to_numpy(self, **kwargs):
        """
        Convert the object stored in this partition to a NumPy array.

        Returns
        -------
            A NumPy array.
        """
        return self.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).get()

    def mask(self, row_indices, col_indices):
        if (
            (isinstance(row_indices, slice) and row_indices == slice(None))
            or (
                not isinstance(row_indices, slice)
                and self._length_cache is not None
                and len(row_indices) == self._length_cache
            )
        ) and (
            (isinstance(col_indices, slice) and col_indices == slice(None))
            or (
                not isinstance(col_indices, slice)
                and self._width_cache is not None
                and len(col_indices) == self._width_cache
            )
        ):
            return self.__copy__()

        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        if not isinstance(row_indices, slice):
            new_obj._length_cache = len(row_indices)
        if not isinstance(col_indices, slice):
            new_obj._width_cache = len(col_indices)
        return new_obj

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        return PandasOnScaleoutFramePartition(
            scaleout.put(obj), len(obj.index), len(obj.columns)
        )

    @classmethod
    def preprocess_func(cls, func):
        """Put a callable function into the plasma store for use in `apply`.

        Args:
            func: The function to preprocess.

        Returns:
            A ray.ObjectRef.
        """
        return scaleout.put(func)

    def length(self):
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = get_index_and_columns.remote(
                    self.oid
                )
        if scaleout.is_object_ref(self._length_cache):
            self._length_cache = scaleout.get(self._length_cache)
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = get_index_and_columns.remote(
                    self.oid
                )
        if scaleout.is_object_ref(self._width_cache):
            self._width_cache = scaleout.get(self._width_cache)
        return self._width_cache

    def ip(self):
        if self._ip_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._ip_cache = self.apply(lambda df: df)._ip_cache
        if scaleout.is_object_ref(self._ip_cache):
            self._ip_cache = scaleout.get(self._ip_cache)
        return self._ip_cache

    @classmethod
    def _length_extraction_fn(cls):
        return length_fn_pandas

    @classmethod
    def _width_extraction_fn(cls):
        return width_fn_pandas

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())


@scaleout.remote(num_returns=2)
def get_index_and_columns(df):
    return len(df.index), len(df.columns)


@scaleout.remote(num_returns=4)
def apply_func(partition, func, *args, **kwargs):  # pragma: no cover
    try:
        result = func(partition, *args, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
    # don't want the error to propagate to the user, and we want to avoid copying unless
    # we absolutely have to.
    except ValueError:
        result = func(partition.copy(), *args, **kwargs)
    return (
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
        scaleout.get_ip(),
    )


@scaleout.remote(num_returns=4)
def apply_list_of_funcs(funcs, partition):  # pragma: no cover
    def deserialize(obj):
        if scaleout.is_object_ref(obj):
            return scaleout.get(obj)
        elif isinstance(obj, (tuple, list)) and any(
            scaleout.is_object_ref(o) for o in obj
        ):
            return scaleout.get(list(obj))
        elif isinstance(obj, dict) and any(
            scaleout.is_object_ref(val) for val in obj.values()
        ):
            return dict(zip(obj.keys(), scaleout.get(list(obj.values()))))
        else:
            return obj

    for func, args, kwargs in funcs:
        func = deserialize(func)
        args = deserialize(args)
        kwargs = deserialize(kwargs)
        try:
            partition = func(partition, *args, **kwargs)
        except ValueError:
            partition = func(partition.copy(), *args, **kwargs)

    return (
        partition,
        len(partition) if hasattr(partition, "__len__") else 0,
        len(partition.columns) if hasattr(partition, "columns") else 0,
        scaleout.get_ip(),
    )
