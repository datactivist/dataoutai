import asyncio
import functools
import json

import regex as re


def dump_to_json(
    file_path: str,
    data: dict,
    encoding: str = "utf8",
    indent: int = 2,
    ensure_ascii: bool = False,
):
    """
    Dumps a Python dict to a JSON file
    :param file_path: The output file path
    :param data: The data to dump
    :param encoding: Encoding to use when writing the file
    :param indent: Ident size for the JSON
    :param ensure_ascii: See the json library documentation for more info
    """
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, fp=f, indent=indent, ensure_ascii=ensure_ascii)


def run_in_executor(executor):
    """
    Pass an executor to the function decorator.
    :param executor: The executor in which we want to run the wrapped function.
    :return: The output of the wrapped function.
    """

    def function_wrapper(f):
        """
        Decorator to call the function in the executor.
        :param f: The function to run.
        :return: The function output.
        """

        @functools.wraps(f)
        def task_to_run(*args, **kwargs):
            """
            Runs the function in the executor.
            :param args: Function arguments.
            :param kwargs: Named function arguments.
            :return: The function output.
            """
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
