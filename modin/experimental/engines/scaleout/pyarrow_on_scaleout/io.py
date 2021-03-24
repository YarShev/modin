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

from modin.backends.pyarrow.query_compiler import PyarrowQueryCompiler
from modin.experimental.engines.scaleout.generic.io import ScaleoutIO
from modin.experimental.engines.scaleout.pyarrow_on_scaleout.frame.data import (
    PyarrowOnScaleoutFrame,
)
from modin.experimental.engines.scaleout.pyarrow_on_scaleout.frame.partition import (
    PyarrowOnScaleoutFramePartition,
)
from modin.backends.pyarrow.parsers import PyarrowCSVParser
from modin.experimental.engines.scaleout.task_wrapper import ScaleoutTask
from modin.engines.base.io import CSVDispatcher


class PyarrowOnScaleoutCSVDispatcher(ScaleoutTask, PyarrowCSVParser, CSVDispatcher):
    frame_cls = PyarrowOnScaleoutFrame
    frame_partition_cls = PyarrowOnScaleoutFramePartition
    query_compiler_cls = PyarrowQueryCompiler


class PyarrowOnScaleoutIO(ScaleoutIO):
    frame_cls = PyarrowOnScaleoutFrame
    frame_partition_cls = PyarrowOnScaleoutFramePartition
    query_compiler_cls = PyarrowQueryCompiler
    csv_reader = PyarrowOnScaleoutCSVDispatcher

    read_parquet_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
