import os

import pytest


@pytest.fixture()
def reset_ports(l=2, a=2, sleep_time=2):
    os.system(f"uhubctl -l {l} -a {a}")
    os.system(f"sleep {sleep_time}s")
