from typing import Optional

import pandas as pd
import autokeras as ak

from mindsdb.integrations.libs.base import BaseMLEngine


class AutokerasHandler(BaseMLEngine):
    """
    Integration with the Autokeras declarative ML library.
    """  # noqa

    name = "autokeras"
