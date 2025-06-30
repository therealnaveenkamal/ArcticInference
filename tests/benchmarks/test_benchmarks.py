import argparse
import json
import multiprocessing
import tempfile
import time
import os

import pytest
import requests
import uvloop
from vllm.entrypoints.openai.api_server import (
    make_arg_parser, run_server, validate_parsed_serve_args)
from vllm.utils import FlexibleArgumentParser

from .benchmark_utils import (ACCURACY_TASKS, PERFORMANCE_TASKS, VLLM_CONFIGS,
                              update_benchmark_summary)


# Set multiprocessing start method to 'spawn' for FlashInfer compatibility
if multiprocessing.get_start_method() != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

# Ensure V1 engine is used
os.environ['VLLM_USE_V1'] = '1'


def _run_lm_eval_process(task_config, model_name, tmpdir, queue):
    """Run lm_eval in a separate process for accuracy benchmarks."""
    try:
        from lm_eval import evaluator
        from lm_eval.utils import handle_non_serializable, make_table

        result = evaluator.simple_evaluate(
            model="local-completions",
            model_args={
                "model": model_name,
                "base_url": "http://localhost:8000/v1/completions",
                "num_concurrent": 256,
            },
            **task_config,
        )
        print(make_table(result))

        tmpfile = f"{tmpdir}/result.json"
        with open(tmpfile, "w") as f:
            json.dump(result, f, indent=4, default=handle_non_serializable)
        
        # Send back the temporary file path
        queue.put(tmpfile)
    except Exception as exc:
        # If an exception occurs, put it in the queue to be raised later
        queue.put(exc)


def _run_vllm_server(args):
    """Run vLLM server in a separate process."""
    uvloop.run(run_server(args))


@pytest.fixture(scope="module", params=list(VLLM_CONFIGS.keys()))
def vllm_server(request):
    """
    Fixture to start the OpenAI API server for testing.
    """
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    args = parser.parse_args([])
    args.disable_log_requests = True
    args.disable_uvicorn_access_log = True

    for key, value in VLLM_CONFIGS[request.param].items():
        setattr(args, key, value)

    # Set environment variable for attention backend if specified
    if args.attention_backend:
        os.environ['VLLM_ATTENTION_BACKEND'] = args.attention_backend
        print(f"Set VLLM_ATTENTION_BACKEND to {args.attention_backend}")

    validate_parsed_serve_args(args)

    # Start server process
    process = multiprocessing.Process(target=_run_vllm_server, args=(args,))
    process.start()

    print("Waiting for server to start...")
    timeout = 1800
    interval = 5
    start = time.time()
    while True:
        try:
            r = requests.get("http://localhost:8000/v1/models")
            if r.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        if not process.is_alive():
            raise RuntimeError("Server process terminated unexpectedly")
        if time.time() - start > timeout:
            raise TimeoutError(f"Server didn't start after {timeout} seconds")
        time.sleep(interval)
    print("Server process started")

    yield request.param, args

    # Stop server process
    print("Terminating server process")
    if process.is_alive():
        process.terminate()
        process.join()
    print("Server process terminated")

    # Clean up environment variable
    if args.attention_backend:
        if 'VLLM_ATTENTION_BACKEND' in os.environ:
            del os.environ['VLLM_ATTENTION_BACKEND']


@pytest.mark.parametrize("task_name", list(PERFORMANCE_TASKS.keys()))
def test_performance(request, vllm_server, task_name):
    from vllm.benchmarks.serve import add_cli_args, main

    config_name, vllm_args = vllm_server
    task = PERFORMANCE_TASKS[task_name]

    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(["--model", vllm_args.model])

    with tempfile.TemporaryDirectory() as tmpdir:
        args.save_result = True
        args.result_dir = str(tmpdir)
        args.result_filename = "result.json"

        for key, value in task.config.items():
            setattr(args, key, value)

        main(args)

        with open(f"{tmpdir}/result.json", "r") as f:
            result = json.load(f)

    benchmark_result_dir = request.config.option.benchmark_result_dir
    if benchmark_result_dir is not None:
        result_path = (benchmark_result_dir / "performance" /
                       f"{config_name}-{task_name}.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    metrics = {name: key(result) if callable(key) else result[key]
               for name, key in task.metrics.items()}
    update_benchmark_summary(config_name, task_name, metrics)


@pytest.mark.parametrize("task_name", list(ACCURACY_TASKS.keys()))
def test_accuracy(request, vllm_server, task_name):

    config_name, vllm_args = vllm_server
    task = ACCURACY_TASKS[task_name]

    assert len(task.config["tasks"]) == 1, \
        "Accuracy benchmarks should only have one task configured"

    q = multiprocessing.Queue()

    with tempfile.TemporaryDirectory() as tmpdir:
        process = multiprocessing.Process(
            target=_run_lm_eval_process,
            args=(task.config, vllm_args.model, tmpdir, q)
        )
        process.start()
        r = q.get()
        process.join()
        if isinstance(r, Exception):
            raise r
        tmpfile = r
        with open(tmpfile, "r") as f:
            result = json.load(f)

    benchmark_result_dir = request.config.option.benchmark_result_dir
    if benchmark_result_dir is not None:
        result_path = (benchmark_result_dir / "accuracy" /
                       f"{config_name}-{task_name}.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    result = result["results"][task.config["tasks"][0]]
    metrics = {name: key(result) if callable(key) else result[key]
               for name, key in task.metrics.items()}
    update_benchmark_summary(config_name, task_name, metrics)
