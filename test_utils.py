import os


def reset_ports(l=2, a=2, sleep_time=2):
    def do_reset(fn):
        def wrapped_do_reset(*args):
            os.system(f"uhubctl -l {l} -a {a}")
            os.system(f"sleep {sleep_time}s")
            fn(*args)

        return wrapped_do_reset

    return do_reset
