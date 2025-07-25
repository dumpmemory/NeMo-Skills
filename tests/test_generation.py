# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import json
import os

# running most things through subprocess since that's how it's usually used
import subprocess
from pathlib import Path

import pytest

from nemo_skills.evaluation.metrics import ComputeMetrics

DATA_TO_TEST = []
template_folder = Path(__file__).parents[1] / 'nemo_skills' / 'prompt' / 'template'
prompt_templates = [f[:-5] for f in os.listdir(template_folder) if f.endswith('.yaml')]

for dataset, split in [('gsm8k', 'train'), ('gsm8k', 'test'), ('math-500', 'test')]:
    DATA_TO_TEST.append((dataset, split))


@pytest.mark.parametrize("dataset,split", DATA_TO_TEST)
def test_generation_dryrun_llama(dataset, split):
    """Testing the default prompts for each dataset."""
    prompt_template = "llama3-instruct"
    extra_args = importlib.import_module(f'nemo_skills.dataset.{dataset}').GENERATION_ARGS
    prompt_config = importlib.import_module(f'nemo_skills.dataset.{dataset}').PROMPT_CONFIG
    cmd = (
        "python nemo_skills/inference/generate.py "
        f"    ++output_file=./test.jsonl "
        f"    ++prompt_template={prompt_template} "
        f"    ++prompt_config={prompt_config} "
        f"    ++input_file=./nemo_skills/dataset/{dataset}/{split}.jsonl "
        f"    ++server.server_type=nemo "
        f"    ++dry_run=True "
        f"    {extra_args} "
    )
    subprocess.run(cmd, shell=True, check=True)


@pytest.mark.parametrize("prompt_template", prompt_templates)
def test_generation_dryrun_gsm8k(prompt_template):
    """Testing that each template can work with a single dataset."""
    dataset = "gsm8k"
    split = "test"
    cmd = (
        "python nemo_skills/inference/generate.py "
        f"    ++output_file=./test.jsonl "
        f"    ++prompt_template={prompt_template} "
        f"    ++prompt_config=generic/math "
        f"    ++input_file=./nemo_skills/dataset/{dataset}/{split}.jsonl "
        f"    ++server.server_type=nemo "
        f"    ++dry_run=True "
    )
    subprocess.run(cmd, shell=True, check=True)


def test_eval_mtbench_api(tmp_path):
    if not os.getenv('NVIDIA_API_KEY'):
        pytest.skip("Define NVIDIA_API_KEY to run this test")

    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=meta/llama-3.1-8b-instruct "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=mt-bench "
        f"    --output_dir={tmp_path} "
        f"    --extra_eval_args=\"++eval_config.use_batch_api=False "
        f"                        ++eval_config.judge_model='meta/llama-3.1-8b-instruct' "
        f"                        ++eval_config.base_url='https://integrate.api.nvidia.com/v1'\" "
        f"    ++max_samples=2 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark='mt-bench').compute_metrics(
        [f"{tmp_path}/eval-results/mt-bench/output.jsonl"],
    )["_all_"]["pass@1"]

    # not having other categories since we just ran with 2 samples
    assert metrics['average'] >= 5
    assert metrics['average_turn1'] >= 5
    assert metrics['average_turn2'] >= 5
    assert metrics['writing_turn1'] >= 5
    assert metrics['writing_turn2'] >= 5
    assert metrics['missing_rating_turn1'] < 2
    assert metrics['missing_rating_turn2'] < 2
    assert metrics['num_entries'] == 2


@pytest.mark.parametrize("format", ["list", "dict"])
def test_generate_openai_format(tmp_path, format):
    if not os.getenv('NVIDIA_API_KEY'):
        pytest.skip("Define NVIDIA_API_KEY to run this test")

    cmd = (
        f"ns generate "
        f"    --server_type=openai "
        f"    --model=meta/llama-3.1-8b-instruct "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --input_file=/nemo_run/code/tests/data/openai-input-{format}.test "
        f"    --output_dir={tmp_path} "
        f"    ++prompt_format=openai "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that output exists and has the expected format
    with open(f"{tmp_path}/output.jsonl") as fin:
        data = [json.loads(line) for line in fin.readlines()]
    assert len(data) == 2
    assert len(data[0]['generation']) > 0
    assert len(data[1]['generation']) > 0
