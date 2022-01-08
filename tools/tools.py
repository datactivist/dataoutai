import asyncio
import functools
import regex as re


def run_in_executor(executor):
    def function_wrapper(f):
        @functools.wraps(f)
        def task_to_run(*args, **kwargs):
            loop = asyncio.get_running_loop()
            return loop.run_in_executor(executor, lambda: f(*args, **kwargs))

        return task_to_run

    return function_wrapper


def remove_xml_tags(string: str):
    """
    Clean str of all XML tags or \\n \\t \\r etc ...
    :param string: String to clean
    """
    spacing = "\n|\t|\r"
    return re.sub("<[^<]*?/?>|" + spacing, "", string)
