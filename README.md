# Oobabooga Benchmark Util

A simple utility for benchmarking LLM performance using the oobabooga text generation Web UI API.

The goal is provide a simple way to perform repeatable performance tests with the oobabooga web UI.  The util can be used to track and check performance of different models, hardware configurations, software configurations (ex. GPU driver versions), etc.

## Basic Usage

```bash
python benchmark.py --infile prompt.txt
```

This will send the text contained in the prompt.txt file to the oobabooga API on the local machine, report the performance metrics (ex. tokens generated per second), save the generated output and produce a CSV of benchmark results.


## Basic settings

#### Settings
| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--infile [INFILE [INFILE ...]]`           | file(s) to read for 'instruct' input. |
| `--outdir OUTDIR`                          | Optional. Output directory for generated text and benchmark data. Default: 'output/' |
| `--host HOST`                              | Optional. Host name or IP address of the OobaBooga server, default = localhost. |
| `--api-streaming-port API_STREAMING_PORT`  | Optional. Port number for the streaming API of the OobaBooga server, default = 5005 |
| ` --api-blocking-port API_BLOCKING_PORT`   | Optional. Port number for the blocking API of the OobaBooga server, default = 5000 |
| `--seed SEED`                              | Optional. Seed value to use for generation, default = -1 (randomize per run) |
| `--max_history_tokens MAX_HISTORY_TOKENS`  | Optional. Max number of history tokens to send along with original prommpt on successive generations, default = 1200 |
| `--max_total_tokens MAX_TOTAL_TOKENS`      | Optional. Max total tokens to generate. Will stop sending requests when this limit is exceeded even if EOS token is not received. Default = 2048 |
| `--tokens_per_gen TOKENS_PER_GEN`          | Optional. Number of tokens to generate on each request. Default = 100 |
| `--temperature TEMPERATURE`                | Optional. Temperature, default = 0.9 |
|`--top_p TOP_P`                             | Optional. top_p, default = 0.9 |
|`--top_k TOP_K`                             | Optional. top_k, default = 100 |
|`--typical_p TYPICAL_P`                     | Optional. typical_p, default = 1 |
|`--epsilon_cutoff EPSILON_CUTOFF`           | Optional. epsilon_cutoff, default = 0 |
|`--eta_cutoff ETA_CUTOFF`                   | Optional. eta_cutoff, default = 0 |
|`--tfs TFS`                                 | Optional. tfs, default = 1 |
|`--top_a TOP_A`                             | Optional. top_a, default = 0 |
|`--repetition_penalty REPETITION_PENALTY`   | Optional. repetition_penalty, default = 1.15 |

## Example Usage

```bash
python benchmark.py --host 192.168.0.10 --userinfo "AMD 5800X3D, 64GB RAM, nVidia 4090" --infile prompt.txt
```

```bash
python benchmark.py --host 192.168.0.10 --api-streaming-port 6500 --seed 600753398 --tokens_per_gen 250 --infile prompt01.txt prompt02.txt prompt03.txt
```

```bash
# Shell Globbing Also works
python benchmark.py --host 192.168.0.10 --infile *.txt
```

> **Note**
> Since the benchmarking util also saves the generated output in addition to the performance metrics, it can be used to generate batches of output with no additional interaction.


```bash
python benchmark.py --host 192.168.0.10 --infile mystories*.txt mypoems*.txt mysongs*.txt
```

## Credits

- Oobabooga Text Generation Web UI: https://github.com/oobabooga/text-generation-webui

