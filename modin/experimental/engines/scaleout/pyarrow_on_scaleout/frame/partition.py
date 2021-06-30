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
from modin.experimental.engines.scaleout.pandas_on_scaleout.frame.partition import (
    PandasOnScaleoutFramePartition,
)

import scaleout
import pyarrow


class PyarrowOnScaleoutFramePartition(PandasOnScaleoutFramePartition):
    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get().to_pandas()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        return PyarrowOnScaleoutFramePartition(
            scaleout.put(pyarrow.Table.from_pandas(obj))
        )

    @classmethod
    def _length_extraction_fn(cls):
        return lambda table: table.num_rows

    @classmethod
    def _width_extraction_fn(cls):
        return lambda table: table.num_columns - (1 if "index" in table.columns else 0)

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())
