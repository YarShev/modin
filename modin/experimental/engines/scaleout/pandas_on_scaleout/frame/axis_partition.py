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

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnScaleoutFramePartition

import scaleout


class PandasOnScaleoutFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks, get_ip=False):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]
        if get_ip:
            self.list_of_ips = [obj._ip_cache for obj in list_of_blocks]

    partition_type = PandasOnScaleoutFramePartition

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        lengths = kwargs.get("_lengths", None)
        return deploy_remote_func.options(
            num_returns=num_splits * 4 if lengths is None else len(lengths) * 4
        ).remote(
            PandasFrameAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            kwargs,
            maintain_partitioning,
            *partitions,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, other_shape, kwargs, *partitions
    ):
        return deploy_remote_func.options(num_returns=num_splits * 4).remote(
            PandasFrameAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            other_shape,
            kwargs,
            *partitions,
        )

    def _wrap_partitions(self, partitions):
        return [
            self.partition_type(object_id, length, width, ip)
            for (object_id, length, width, ip) in zip(*[iter(partitions)] * 4)
        ]


class PandasOnScaleoutFrameColumnPartition(PandasOnScaleoutFrameAxisPartition):
    """The column partition implementation for Scaleout. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 0


class PandasOnScaleoutFrameRowPartition(PandasOnScaleoutFrameAxisPartition):
    """The row partition implementation for Scaleout. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 1


@scaleout.remote
def deploy_remote_func(func, *args):  # pragma: no cover
    """
    Run a function on a remote partition.

    Parameters
    ----------
    func : callable
        The function to run.

    Returns
    -------
        The result of the function `func`.

    Notes
    -----
    Ray functions are not detected by codecov (thus pragma: no cover)
    """
    result = func(*args)
    ip = scaleout.get_ip()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
