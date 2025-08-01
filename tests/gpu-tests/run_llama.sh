# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
# model needs to be inside /mnt/datadrive/nemo-skills-test-data/Meta-Llama-3.1-8B-Instruct
# if you need to place it in a different location, modify test-local.yaml config
# example: HF_TOKEN=<> ./tests/gpu-tests/run.sh
set -e

export NEMO_SKILLS_TEST_HF_MODEL=/mnt/datadrive/nemo-skills-test-data/Meta-Llama-3.1-8B-Instruct
export NEMO_SKILLS_TEST_MODEL_TYPE=llama

# first running the conversion tests
pytest tests/gpu-tests/test_convert.py -k test_hf_trtllm_conversion -s -x
export NEMO_SKILLS_TEST_TRTLLM_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/conversion/hf-to-trtllm/model
pytest tests/gpu-tests/test_convert.py -k test_hf_nemo_conversion -s -x
export NEMO_SKILLS_TEST_NEMO_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/conversion/hf-to-nemo/model
pytest tests/gpu-tests/test_convert.py -k test_hf_megatron_conversion -s -x
export NEMO_SKILLS_TEST_MEGATRON_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/conversion/hf-to-megatron/model
pytest tests/gpu-tests/test_convert.py -k test_nemo_hf_conversion -s -x
# using the back-converted model to check that it's reasonable
export NEMO_SKILLS_TEST_HF_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/conversion/nemo-to-hf/model

# generation/evaluation tests
pytest tests/gpu-tests/test_logprobs.py -s -x
pytest tests/gpu-tests/test_eval.py -s -x
pytest tests/gpu-tests/test_generate.py -s -x
pytest tests/gpu-tests/test_judge.py -s -x
pytest tests/gpu-tests/test_run_cmd_llm_infer.py -s -x
pytest tests/gpu-tests/test_contamination.py -s -x

# for sft we are using the tiny random model to run much faster
ns run_cmd --cluster test-local --config_dir tests/gpu-tests --container nemo \
    python /nemo_run/code/tests/gpu-tests/make_tiny_llm.py --model_type $NEMO_SKILLS_TEST_MODEL_TYPE

# converting the model through test
export NEMO_SKILLS_TEST_HF_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/tiny-model-hf
pytest tests/gpu-tests/test_convert.py -k test_hf_nemo_conversion -s -x
# training tests
export NEMO_SKILLS_TEST_NEMO_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/conversion/hf-to-nemo/model
pytest tests/gpu-tests/test_train.py -s -x
