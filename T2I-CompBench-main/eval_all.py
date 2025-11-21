import os
import json
import subprocess
from pathlib import Path

# Root paths
ROOT = Path(__file__).absolute().parent
EXAMPLES = (ROOT / "../examples").resolve()
SAMPLES = EXAMPLES / "samples"
LABELS = EXAMPLES / "labels"

# ===== Helper =====
def run_cmd(cmd: str):
    print(f"\n\n===== Running: {cmd} =====\n")
    status = subprocess.run(cmd, shell=True)
    if status.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def ensure_dirs():
    (LABELS / "depth").mkdir(parents=True, exist_ok=True)
    (LABELS / "depth" / "samples").mkdir(parents=True, exist_ok=True)

def fix_depth_structure():
    """Fix wrong depth map directory from 3D evaluator (move depth/examples/samples -> depth/samples)."""
    wrong_dir = LABELS / "depth" / "examples" / "samples"
    correct_dir = LABELS / "depth" / "samples"

    if wrong_dir.exists():
        for f in wrong_dir.iterdir():
            f.rename(correct_dir / f.name)
        # remove the extra 'examples' directory
        os.system(f"rm -rf {}".format(LABELS / "depth" / "examples"))
        print("✔ Fixed depth directory structure.")

# ===== Single metrics =====
def eval_vqa(results: dict):
    """BLIP-VQA (attribute binding)."""
    run_cmd("python BLIPvqa_eval/BLIP_vqa.py --out_dir=../examples")
    # 你目前的 BLIP_vqa.py 里已经会在 examples/ 下写 vqa_result.json
    res_path = EXAMPLES / "vqa_result.json"
    if res_path.exists():
        results["VQA"] = json.load(open(res_path))
    else:
        results["VQA"] = "No result produced."

def eval_2d_spatial(results: dict):
    """UniDet 2D spatial evaluation."""
    run_cmd("python UniDet_eval/2D_spatial_eval.py")
    r = EXAMPLES / "annotation_spatial2d" / "vqa_result.json"
    if r.exists():
        results["2D_Spatial"] = json.load(open(r))
    else:
        results["2D_Spatial"] = "No results."

def eval_numeracy(results: dict):
    """UniDet numeracy evaluation."""
    run_cmd("python UniDet_eval/numeracy_eval.py")
    r = EXAMPLES / "annotation_num" / "vqa_result.json"
    if r.exists():
        results["Numeracy"] = json.load(open(r))
    else:
        results["Numeracy"] = "No results."

def eval_3d_spatial(results: dict):
    """UniDet 3D spatial evaluation (depth + detection)."""
    ensure_dirs()
    run_cmd("python UniDet_eval/3D_spatial_eval.py")
    fix_depth_structure()
    r = EXAMPLES / "annotation_spatial3d" / "vqa_result.json"
    if r.exists():
        results["3D_Spatial"] = json.load(open(r))
    else:
        results["3D_Spatial"] = "No results."

def eval_clip_similarity(results: dict):
    """CLIPScore for non-spatial relationships."""
    # 官方 README 的用法：
    # outpath="examples/"; python CLIPScore_eval/CLIP_similarity.py --outpath=${outpath}
    run_cmd("python CLIPScore_eval/CLIP_similarity.py --outpath=../examples")
    r = EXAMPLES / "clip_result.json"
    if r.exists():
        results["CLIP_Similarity"] = json.load(open(r))
    else:
        results["CLIP_Similarity"] = "No results."

# ===== 3-in-1 (official combined metric) =====
def eval_three_in_one(results: dict):
    """Run official 3-in-1 evaluator (attribute + spatial combo)."""
    # 对应 README 里 4. 3-in-1 for Complex Compositions
    run_cmd("python 3_in_1_eval/3_in_1.py --outpath=../examples")
    r = EXAMPLES / "annotation_3_in_1" / "vqa_result.json"
    if r.exists():
        results["3_in_1"] = json.load(open(r))
    else:
        results["3_in_1"] = "No results."

# ===== MLLM (GPT-4V / GPT-4o etc.) =====
# 可通过环境变量改这三个配置：
#   GPT4V_CATEGORIES="color,shape,texture,spatial,non_spatial,complex"
#   GPT4V_START=0
#   GPT4V_STEP=10
GPT4V_CATEGORIES = os.getenv("GPT4V_CATEGORIES", "complex").split(",")
GPT4V_START = int(os.getenv("GPT4V_START", "0"))
GPT4V_STEP = int(os.getenv("GPT4V_STEP", "1"))

def eval_mllm_gpt4v(results: dict):
    """Run GPT-4V / GPT-4o evaluation via MLLM_eval/gpt4v_eval.py.

    使用前请先：
      1) 在 MLLM_eval/gpt4v_eval.py 里填好 OpenAI API key
      2) （可选）用环境变量调整 GPT4V_CATEGORIES / GPT4V_START / GPT4V_STEP
    """
    mllm_scores = {}
    for cat in GPT4V_CATEGORIES:
        cat = cat.strip()
        if not cat:
            continue
        cmd = (
            f'cd MLLM_eval && '
            f'python MLLM_eval/gpt4v_eval.py '
            f'--category "{cat}" --start {GPT4V_START} --step {GPT4V_STEP}'
        )
        run_cmd(cmd)
        # 输出文件：examples/gpt4v/gpt4v_result_{start}_{step}.json
        out_name = f"gpt4v_result_{GPT4V_START}_{GPT4V_STEP}.json"
        r = EXAMPLES / "gpt4v" / out_name
        if r.exists():
            mllm_scores[cat] = json.load(open(r))
        else:
            mllm_scores[cat] = "No results."

    results["MLLM_GPT4V"] = mllm_scores

def main():
    results = {}

    print("🚀 Starting T2I-CompBench full evaluation...")

    # 1. 基础自动指标
    eval_vqa(results)
    eval_2d_spatial(results)
    eval_numeracy(results)
    eval_3d_spatial(results)
    eval_clip_similarity(results)

    # 2. 官方 3-in-1 综合指标
    eval_three_in_one(results)

    # 3. 多模态大模型评测（GPT-4V / GPT-4o）
    try:
        eval_mllm_gpt4v(results)
    except Exception as e:
        print(f"⚠️ MLLM evaluation failed: {e}")
        results["MLLM_GPT4V"] = f"Failed: {e}"

    # 总结到一个 json 里
    summary_path = ROOT / "final_eval_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n\n🎉 ALL DONE! Results saved to:")
    print(summary_path)

if __name__ == "__main__":
    main()