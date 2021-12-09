import asyncio
import functools


def run_in_executor(executor):
    def function_wrapper(f):
        @functools.wraps(f)
        def task_to_run(*args, **kwargs):
            loop = asyncio.get_running_loop()
            return loop.run_in_executor(executor, lambda: f(*args, **kwargs))

        return task_to_run

    return function_wrapper
