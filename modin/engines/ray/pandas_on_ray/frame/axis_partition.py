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

from modin.config import EnablePartitionIPs
from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnRayFramePartition

import ray


if EnablePartitionIPs.get():
    from ray.services import get_node_ip_address


class PandasOnRayFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks, bind_ip=False):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]
        if bind_ip:
            if EnablePartitionIPs.get():
                self.list_of_ips = [obj.ip for obj in list_of_blocks]
            else:
                raise ValueError(
                    "Passed `bind_ip=True` but `MODIN_ENABLE_PARTITIONS_API` env var was not exported."
                )

    partition_type = PandasOnRayFramePartition
    instance_type = ray.ObjectID

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        lengths = kwargs.get("_lengths", None)
        factor = 4 if EnablePartitionIPs.get() else 3
        return deploy_ray_func._remote(
            args=(
                PandasFrameAxisPartition.deploy_axis_func,
                axis,
                func,
                num_splits,
                kwargs,
                maintain_partitioning,
            )
            + tuple(partitions),
            num_returns=num_splits * factor
            if lengths is None
            else len(lengths) * factor,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, other_shape, kwargs, *partitions
    ):
        factor = 4 if EnablePartitionIPs.get() else 3
        return deploy_ray_func._remote(
            args=(
                PandasFrameAxisPartition.deploy_func_between_two_axis_partitions,
                axis,
                func,
                num_splits,
                len_of_left,
                other_shape,
                kwargs,
            )
            + tuple(partitions),
            num_returns=num_splits * factor,
        )

    def _wrap_partitions(self, partitions):
        if EnablePartitionIPs.get():
            return [
                self.partition_type(object_id, length, width, ip)
                for (object_id, length, width, ip) in zip(*[iter(partitions)] * 4)
            ]
        else:
            return [
                self.partition_type(object_id, length, width)
                for (object_id, length, width) in zip(*[iter(partitions)] * 3)
            ]


class PandasOnRayFrameColumnPartition(PandasOnRayFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 0


class PandasOnRayFrameRowPartition(PandasOnRayFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 1


@ray.remote
def deploy_ray_func(func, *args):  # pragma: no cover
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
    if EnablePartitionIPs.get():
        ip = get_node_ip_address()
        if isinstance(result, pandas.DataFrame):
            return result, len(result), len(result.columns), ip
        elif all(isinstance(r, pandas.DataFrame) for r in result):
            return [i for r in result for i in [r, len(r), len(r.columns), ip]]
        else:
            return [i for r in result for i in [r, None, None, ip]]
    else:
        if isinstance(result, pandas.DataFrame):
            return result, len(result), len(result.columns)
        elif all(isinstance(r, pandas.DataFrame) for r in result):
            return [i for r in result for i in [r, len(r), len(r.columns)]]
        else:
            return [i for r in result for i in [r, None, None]]
