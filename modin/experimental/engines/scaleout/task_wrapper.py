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

import scaleout


@scaleout.remote
def deploy_remote_func(func, args):  # pragma: no cover
    return func(**args)


class ScaleoutTask:
    @classmethod
    def deploy(cls, func, num_returns, kwargs):
        return deploy_remote_func.options(num_returns=num_returns).remote(func, kwargs)

    @classmethod
    def materialize(cls, obj_id):
        return scaleout.get(obj_id)