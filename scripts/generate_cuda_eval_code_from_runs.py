import argparse
import os
from litellm import completion
from dotenv import load_dotenv
from string import Template

load_dotenv()

RUNS_ROOT = "runs"
OUTPUT_ROOT = "cuda_eval_code"
LEVEL_PROBLEMS = {
    1: range(1, 101),
    2: range(1, 101),
    3: range(1, 51),
}

MODEL_NAME = "anthropic/claude-sonnet-4-5-20250929"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--level", type=int, required=True)
    return parser.parse_args()


def extract_kernel_code(
    run_name: str, level: int, problem_id: int, sample_id: int = 0
) -> str:
    kernel_path = os.path.join(
        RUNS_ROOT,
        run_name,
        f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py",
    )

    if not os.path.exists(kernel_path):
        print(f"Warning: Kernel file not found at {kernel_path}")
        return None, None, None

    code = ""
    with open(kernel_path, "r") as f:
        code = f.read()

    ############ #define macro code     ############
    macro_code = ""
    for line in code.split("\n"):
        if line.startswith("#define"):
            macro_code += line + "\n"

    ############ __global__ kernel code ############

    kernel_code = ""
    in_kernel = False
    for line in code.split("\n"):
        if line.startswith("__global__"):
            in_kernel = True
            kernel_code += line + "\n"
        if in_kernel:
            kernel_code += line + "\n"
        if line.startswith("}"):
            kernel_code += line + "\n"
            in_kernel = False

    ############ torch::Tensor torch entry code ####

    entry_code = ""
    in_entry = False
    for line in code.split("\n"):
        if line.startswith("torch::Tensor"):
            in_entry = True
            entry_code += line + "\n"
        if in_entry:
            entry_code += line + "\n"
        if line.startswith("}"):
            entry_code += line + "\n"
            in_entry = False

    return macro_code, kernel_code, entry_code


# one-shot template for generate c++ entry code
one_shot_template = Template(f"""You are a CUDA expert. Your task is to generate a c++ entry code for the given kernel code and torch::Tensor entry code. You should follow the following rules:
- You should use the original kernel code and torch::Tensor entry code as a reference.
- You should generate the entry code that is compatible with the torch::Tensor entry code.
- You should write the entry code with the original kernel code
- Do not change the original kernel code.
- Do not change the original macro code.
- Do not change the original #include code.
- Do not change the original #define code.


You are given the following kernel code and torch::Tensor entry code:
```cpp
${macro_code}
${kernel_code}
${entry_code}
```

Your example entry code:
```cpp
__global__ void leaky_relu_kernel_ori(const float* x, float* y, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = (x[idx] > 0.0) ? x[idx] : x[idx] * negative_slope;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
    {
    int  size = in_elems;
    float negative_slope = 0.01;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    leaky_relu_kernel_ori<<<num_blocks, block_size>>>(input, output,negative_slope, size);
}
```
""")


def make_prompt(macro_code: str, kernel_code: str, entry_code: str) -> str:
    one_shot_prompt = one_shot_template.substitute(
        macro_code=macro_code, kernel_code=kernel_code, entry_code=entry_code
    )
    return one_shot_prompt


def main():
    args = parse_args()
    run_name = args.run_name
    level = args.level
    sample_id = 0
    output_dir = os.path.join(OUTPUT_ROOT, run_name, f"level_{level}")
    os.makedirs(output_dir, exist_ok=True)
    for problem_id in LEVEL_PROBLEMS[level]:
        macro_code, kernel_code, entry_code = extract_kernel_code(
            run_name, level, problem_id, sample_id
        )
        if kernel_code is None or entry_code is None:
            continue
        prompt = make_prompt(macro_code, kernel_code, entry_code)
        response = completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
            api_key=ANTHROPIC_API_KEY,
        )
        eval_code = response.choices[0].message.content
        output_dir_problem = os.path.join(output_dir, f"problem_{problem_id}")
        os.makedirs(output_dir_problem, exist_ok=True)
        with open(os.path.join(output_dir_problem, f"tmp_ori.cu"), "w") as f:
            f.write(eval_code)
        return


if __name__ == "__main__":
    main()
