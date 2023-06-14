import asyncio
import json
import sys
import argparse
import os
import time
import requests
import random
import csv
from datetime import datetime, timezone

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

DEFAULT_OUTDIR='output/'
EXECUTION_TIME=time.time_ns() // 1_000_000_000
EOS_TOKEN = "</s>"

DEFAULT_TEMPERATURE=0.9
DEFAULT_TOP_P=0.9
DEFAULT_TOP_K=100
DEFAULT_TYPICAL_P=1
DEFAULT_EPSILON_CUTOFF=0
DEFAULT_ETA_CUTOFF=0
DEFAULT_TFS=1
DEFAULT_TOP_A=0
DEFAULT_REPETITION_PENALTY=1.15


DEFAULT_TOKENS_PER_GEN=100
DEFAULT_MAX_TOTAL_TOKENS=2048
DEFAULT_SEED=-1
DEFAULT_MAX_HISTORY_TOKENS=1200

DEFAULT_HOST="localhost"
DEFAULT_STREAMING_PORT=5005
DEFAULT_BLOCKING_PORT=5000

class BenchmarkData:
    def __init__(self, initial_prompt: str, infilename: str, outfilename: str, csvfilename: str):
        self.initial_prompt = initial_prompt
        self.infilename = infilename
        self.outfilename = outfilename
        self.csvfilename = csvfilename
        self.done = False
        self.history = []

    def append_history(self, tokenText):
        self.history.append(tokenText)

    def history_context(self, max_history_tokens):
        if (self.total_history_tokens() <= max_history_tokens):
            return self.initial_prompt + "".join(list(map(lambda x: x.text, self.history)))
        else:
            truncated_history = []
            tokens_length = 0
            for temp_hist in reversed(self.history):
                if tokens_length + temp_hist.tokens <= max_history_tokens:
                    truncated_history.insert(0, temp_hist)
                    tokens_length += temp_hist.tokens
                else:
                    break
            return self.initial_prompt + "...".join(list(map(lambda x: x.text, truncated_history)))
                
    
    def history_text(self):
        return "".join(list(map(lambda x: x.text, self.history)))
    
    def total_history_tokens(self):
        return sum(list(map(lambda x: x.tokens, self.history)))

    def total_runtime(self):
        return sum(list(map(lambda x: x.runtime, self.history)))

    def average_tokens_per_second(self):
        return 0.0 if len(self.history) == 0 else (self.total_history_tokens() / self.total_runtime())
    
class History:
    def __init__(self, text: str, tokens: int, runtime: float):
        self.text = text
        self.tokens = tokens
        self.runtime = runtime

    def tokens_per_second(self):
        return 0.0 if self.runtime == 0.0 else (self.tokens / self.runtime)

class BenchmarkConfig:
    def __init__(self, temperature: float, top_p: float, top_k: int, typcial_p: int, epsilon_cutoff: float, 
                 eta_cutoff: float, tfs: float, top_a: float, repetition_penalty: float,
                 tokens_per_gen: int, max_total_tokens: int, max_history_tokens: int, seed: int, 
                 host:str, streaming_port: int, blocking_port: int, userinfo: str):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.typical_p = int(typcial_p)
        self.epsilon_cutoff = float(epsilon_cutoff)
        self.eta_cutoff = float(eta_cutoff)
        self.tfs = float(tfs)
        self.top_a = float(top_a)
        self.repetition_penalty = float(repetition_penalty)
        self.tokens_per_gen = int(tokens_per_gen)
        self.max_total_tokens = int(max_total_tokens)
        self.max_history_tokens = int(max_history_tokens)
        self.seed = int(seed)
        self.uri = f'ws://{host}:{streaming_port}/api/v1/stream'
        self.model_uri = f'http://{host}:{blocking_port}/api/v1/model'
        self.userinfo = userinfo

async def run(benchmark_data: BenchmarkData, config: BenchmarkConfig):
    request = {
        'prompt': benchmark_data.history_context(config.max_history_tokens),
        'max_new_tokens': config.tokens_per_gen,
        'do_sample': True,
        'temperature': config.temperature,
        'top_p': config.top_p,
        'typical_p': config.typical_p,
        'epsilon_cutoff': config.epsilon_cutoff,  # In units of 1e-4
        'eta_cutoff': config.eta_cutoff,  # In units of 1e-4
        'tfs': config.tfs,
        'top_a': config.top_a,
        'repetition_penalty': config.repetition_penalty,
        'top_k': config.top_k,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'seed': config.seed,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': False,
        'stopping_strings': []
    }

    async with websockets.connect(config.uri, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))
        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)
            if incoming_data['event'] == 'text_stream':
                yield incoming_data
            elif incoming_data['event'] == 'stream_end':
                tokens = incoming_data['message_num'] - 1
                return



async def generate_response_stream(benchmark_data: BenchmarkData, config: BenchmarkConfig):
    cur_len = 0
    curMessage = ""
    startTime = time.perf_counter_ns()
    async for data in run(benchmark_data, config):
        if data['text'].endswith(EOS_TOKEN) or benchmark_data.total_history_tokens() >= config.max_total_tokens:
            benchmark_data.done = True
            curMessage += data['text'].replace(EOS_TOKEN, "\n\n")
        else:
            curMessage += data['text']
        sys.stdout.flush()  # If we don't flush, we won't see tokens in realtime.
    runtime = (time.perf_counter_ns() - startTime) / 1_000_000 / 1000
    tokens = data['message_num']
    print(f'\tGeneration {len(benchmark_data.history) + 1} done in {runtime:.4f}s ({(tokens / runtime):.2f} tokens/s)')
    benchmark_data.append_history(History(curMessage, tokens, runtime))

def initialize_arg_parser():
    argParser = argparse.ArgumentParser('oobabooga Benchmark')
    argParser.add_argument('--infile', required=True, nargs='*',
                    help="file(s) to read for 'instruct' input")
    argParser.add_argument('--outdir', 
                    help="Optional. Output directory for generated text and benchmark data. Default: 'output/'")
    argParser.add_argument('--host', default=DEFAULT_HOST,
                    help=f"Optional. Host name or IP address of the oobabooga server, default = {DEFAULT_HOST}")
    argParser.add_argument('--api-streaming-port', default=DEFAULT_STREAMING_PORT,
                    help=f"Optional. Port number for the streaming API of the oobabooga server, default = {DEFAULT_STREAMING_PORT}")
    argParser.add_argument('--api-blocking-port', default=DEFAULT_BLOCKING_PORT,
                    help=f"Optional. Port number for the blocking API of the oobabooga server, default = {DEFAULT_BLOCKING_PORT}")
    argParser.add_argument('--seed', default=DEFAULT_SEED,
                    help=f"Optional. Seed value to use for generation, default = {DEFAULT_SEED} (randomize per run)")
    argParser.add_argument('--max_history_tokens', default=DEFAULT_MAX_HISTORY_TOKENS,
                    help=f"Optional. Max number of history tokens to send along with original prommpt on successive generations, default = {DEFAULT_MAX_HISTORY_TOKENS}")
    argParser.add_argument('--max_total_tokens', default=DEFAULT_MAX_TOTAL_TOKENS,
                    help=f"Optional. Max total tokens to generate. Will stop sending requests when this limit is exceeded even if EOS token is not received. Default = {DEFAULT_MAX_TOTAL_TOKENS}")
    argParser.add_argument('--tokens_per_gen', default=DEFAULT_TOKENS_PER_GEN,
                    help=f"Optional. Number of tokens to generate on each request. Default = {DEFAULT_TOKENS_PER_GEN}")
    argParser.add_argument('--temperature', default=DEFAULT_TEMPERATURE,
                    help=f"Optional. Temperature, default = {DEFAULT_TEMPERATURE}")
    argParser.add_argument('--top_p', default=DEFAULT_TOP_P,
                    help=f"Optional. top_p, default = {DEFAULT_TOP_P}")
    argParser.add_argument('--top_k', default=DEFAULT_TOP_K,
                    help=f"Optional. top_k, default = {DEFAULT_TOP_K}")
    argParser.add_argument('--typical_p', default=DEFAULT_TYPICAL_P,
                    help=f"Optional. typical_p, default = {DEFAULT_TYPICAL_P}")
    argParser.add_argument('--epsilon_cutoff', default=DEFAULT_EPSILON_CUTOFF,
                    help=f"Optional. epsilon_cutoff, default = {DEFAULT_EPSILON_CUTOFF}")
    argParser.add_argument('--eta_cutoff', default=DEFAULT_ETA_CUTOFF,
                    help=f"Optional. eta_cutoff, default = {DEFAULT_ETA_CUTOFF}")
    argParser.add_argument('--tfs', default=DEFAULT_TFS,
                    help=f"Optional. tfs, default = {DEFAULT_TFS}")
    argParser.add_argument('--top_a', default=DEFAULT_TOP_A,
                    help=f"Optional. top_a, default = {DEFAULT_TOP_A}")
    argParser.add_argument('--repetition_penalty', default=DEFAULT_REPETITION_PENALTY,
                    help=f"Optional. repetition_penalty, default = {DEFAULT_REPETITION_PENALTY}")

    argParser.add_argument('--userinfo',
                    help=f"Optional. User defined information to include in the benchmark result. For example, can be used to record system info (cpu, gpu, RAM, etc.) Note: Make sure to enclose user info in quotes: \"my custom user info\"")

    return argParser

def initialize_benchmark_data(args):
    benchmarks = []
    outdirname = args.outdir
    if outdirname == None:
        outdirname = DEFAULT_OUTDIR
    if outdirname == None or outdirname.strip() == "":
        outdirname = "./"
    if not os.path.exists(outdirname):
        try:
            os.makedirs(outdirname)
        except PermissionError:
            sys.exit(f'Unable to create output directory {outdirname}.')
    if not os.access(outdirname, os.W_OK):
        print (outdirname)
        sys.exit(f'Unable to open {outdirname} for writing. Please check that the specified output directory exists and you have write permissions to that directory.')

    for infilename in args.infile:
        if not (os.path.exists(infilename) and os.access(infilename, os.R_OK)):
            sys.exit(f'Unable to open {infilename} for reading.')
        try:
            infp = open(infilename)
        except PermissionError:
            sys.exit(f'Unable to open {infilename} for reading.')
        else:
            with infp:
                indata = infp.read()
        outfilename = os.path.join(outdirname, f'{EXECUTION_TIME}_{infilename}')
        csvfilename = f'{os.path.splitext(outfilename)[0]}.csv'
        benchmarks.append(BenchmarkData(initial_prompt=indata, 
                                        infilename=infilename, 
                                        outfilename=outfilename,
                                        csvfilename=csvfilename))

    return benchmarks

def initialize_config(args):
    seed = int(args.seed)
    if seed == -1:
        seed = random.getrandbits(32)
    return BenchmarkConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        typcial_p=args.typical_p,
        epsilon_cutoff=args.epsilon_cutoff,
        eta_cutoff=args.eta_cutoff,
        tfs=args.tfs,
        top_a=args.top_a,
        repetition_penalty=args.repetition_penalty,
        seed=seed,
        max_total_tokens=args.max_total_tokens,
        max_history_tokens=args.max_history_tokens,
        tokens_per_gen=args.tokens_per_gen,
        host=args.host,
        streaming_port=args.api_streaming_port,
        blocking_port=args.api_blocking_port,
        userinfo=args.userinfo
    )

def fetch_model_info(config: BenchmarkConfig):
    response = requests.post(config.model_uri, json={'action': 'info'}, timeout=5)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def output_results(benchmark: BenchmarkData, config: BenchmarkConfig, model: str):
    with open(benchmark.csvfilename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([f'# {config.userinfo}'])
        writer.writerow([f'# Model: {model}'])
        writer.writerow([f'# Seed: {config.seed}'])
        writer.writerow([f'# Benchmarked on: {datetime.fromtimestamp(EXECUTION_TIME, tz=timezone.utc)}'])
        writer.writerow(["Generation", "Tokens", "Tokens/s", "Context", "Runtime (s)"])
        context = 0
        for index, history in enumerate(benchmark.history):
            context += history.tokens
            if context > config.max_history_tokens:
                context = config.max_history_tokens
            writer.writerow([index + 1, history.tokens, f'{history.tokens_per_second():.2f}', context, f'{history.runtime:.4f}'])
        writer.writerow(["Total", benchmark.total_history_tokens(), f'{benchmark.average_tokens_per_second():.2f}', context, f'{benchmark.total_runtime():.4f}' ])




if __name__ == '__main__':
    arg_parser = initialize_arg_parser()
    args = arg_parser.parse_args()
    benchmarks = initialize_benchmark_data(args)
    config = initialize_config(args)

    model_info = fetch_model_info(config)
    if model_info != None:
        model = model_info['result']['model_name']
    else:
        model = 'Unknown'

    if len(benchmarks) == 0:
        sys.exit('No benchmark input files. Nothing to do.')

    if config.userinfo:
        print(f'Benchmarking {config.userinfo}')

    for index, benchmark in enumerate(benchmarks):
        # print(f'{os.path.splitext(benchmark.outfilename)[0]}')
        # continue
        print(f'Input: {benchmark.infilename}')
        while not benchmark.done:
            asyncio.run(generate_response_stream(benchmark, config))
        # for index, history in enumerate(benchmark.history):
        #     print(f'Generation {index}: {history.tokens} tokens in {history.runtime:.2f}s {history.tokensPerSecond():.2f} tokens/s')
        print(f'\tFinished benchmarking {model} with prompt {benchmark.infilename} and seed {config.seed}. Generated {benchmark.total_history_tokens()} tokens over {len(benchmark.history)} generations in {benchmark.total_runtime():.2f}s. Averaging {benchmark.average_tokens_per_second():0.2f} tokens/s')
        output_results(benchmark=benchmark, config=config, model=model)
        
        try:
            outfile = open(benchmark.outfilename, mode = 'w')
            outfile.write(benchmark.history_text())
            outfile.close()
            print(f'\tOutput saved to {benchmark.outfilename}.')
            print(f'\tResults saved to {benchmark.csvfilename}.')
        except PermissionError:
            sys.exit(f'Unable to open {benchmark.outfilename} for writing.')
