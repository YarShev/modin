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

from modin.config import EnablePartitionIPs

import numpy as np
from distributed.client import _get_global_client
from distributed.utils import get_ip


class DaskTask:
    @classmethod
    def deploy(cls, func, num_returns, kwargs):
        client = _get_global_client()
        remote_task_future = client.submit(func, **kwargs)
        if EnablePartitionIPs.get():
            futures = [
                client.submit(lambda l, i: (l[i], get_ip()), remote_task_future, i)
                for i in range(num_returns)
            ]
            futures_with_ips = np.array(
                [
                    [
                        client.submit(lambda l, i, j: l[i][j], futures, i, j)
                        for j in range(2)
                    ]
                    for i in range(num_returns)
                ]
            )
            return list(zip(futures_with_ips[:, 0], futures_with_ips[:, 1]))
        else:
            return [
                client.submit(lambda l, i: l[i], remote_task_future, i)
                for i in range(num_returns)
            ]

    @classmethod
    def materialize(cls, future):
        client = _get_global_client()
        return client.gather(future)
