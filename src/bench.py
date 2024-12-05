import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from time import perf_counter_ns, sleep
from typing import List

from rich.progress import track

logger = logging.getLogger(__name__)


@dataclass
class LogPoint:
    args: tuple
    start_time: int
    end_time: int
    n: int

    @staticmethod
    def csv_header():
        return "start_time,end_time,args,n"

    def to_csv(self):
        return f"{self.start_time},{self.end_time},{self.args},{self.n}"


@dataclass
class Log:
    log_points: List[LogPoint]

    def to_csv(self, path: str):
        with open(path, "w") as f:
            f.write(LogPoint.csv_header() + "\n")
            f.writelines(lp.to_csv() + "\n" for lp in self.log_points)

    def append(self, log_point: LogPoint):
        self.log_points.append(log_point)


def iter_dict(d: dict):
    if not d:
        yield {}
        return

    key, value = d.popitem()

    if not isinstance(value, list):
        value = [value]

    yield from ({key: v, **sub_v} for v in value for sub_v in iter_dict(d))


class ArgSpace:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.arg_gen = iter_dict(kwargs)

    def __len__(self):
        return prod(len(v) for v in self.kwargs.values() if isinstance(v, list))

    def __iter__(self):
        return self.arg_gen

    def __next__(self):
        return next(self.arg_gen)


class BenchMark:
    def __init__(
        self,
        function: Callable,
        arg_generator,
        cooldown_function: Callable = lambda: sleep(5),
    ):
        self.function = function
        self.arg_generator = arg_generator
        self.cooldown_function = cooldown_function

    def run(self, sample_period_sec: int = 30) -> Log:
        """Use timeit to run the benchmark, recording the times at which the function starts and stops.
        This data will then be linked with the power consumption time series to give the energy used per function call.
        """
        log = Log([])
        logger.info("Starting benchmark")

        def format_time(time):
            units = ["ns", "us", "ms"]
            value = [1, 1e3, 1e6]

            for u, v in zip(units, value):
                if time < 1000 * v:
                    return f"{time / v:.2f} {u}"

            return f"{time / 1e9:.2f} s"

        for args in track(
            self.arg_generator,
            description="Benchmarking...",
        ):
            logger.info(f"Running benchmark with args: {args}")

            start_time = perf_counter_ns()
            iters = 0

            while perf_counter_ns() - start_time < sample_period_sec * 1e9:
                self.function(**args)
                iters += 1

            end_time = perf_counter_ns()

            log.append(LogPoint(args, start_time, end_time, iters))
            time = (end_time - start_time) / iters

            logger.info(f"Finished benchmark. Mean time per item: {format_time(time)}")
            logger.info(f"Cooling down")

            self.cooldown_function()

        return log


if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    def fib(n):
        dp = [0, 1]
        for i in range(2, n + 1):
            dp.append(dp[i - 1] + dp[i - 2])

        return dp[n]

    arg_space = ArgSpace(n=[10, 20, 30, 40])

    bm = BenchMark(fib, arg_space)
    log = bm.run(sample_period_sec=5)
    log.to_csv("log.csv")
