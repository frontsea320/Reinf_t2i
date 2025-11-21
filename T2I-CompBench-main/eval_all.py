import os
import json
import subprocess
from pathlib import Path

# Root paths
ROOT = Path(__file__).absolute().parent
# [FIX] 修正路径：examples 在当前目录下，而不是上一级
EXAMPLES = (ROOT / "examples").resolve()
SAMPLES = EXAMPLES / "samples"
LABELS = EXAMPLES / "labels"

# ===== Helper =====
def run_cmd(cmd: str, cwd: Path = ROOT):
    """
    在指定目录(cwd)下执行命令。
    """
    print(f"\n\n===== Running: {cmd} (in {cwd}) =====\n")
    # shell=True 允许我们使用 cd 命令 (虽然 subprocess.run 的 cwd 参数更好)
    # 这里为了兼容性，直接使用 cwd 参数控制工作目录
    status = subprocess.run(cmd, shell=True, cwd=str(cwd))
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
        os.system(f"rm -rf {LABELS / 'depth' / 'examples'}")
        print("✔ Fixed depth directory structure.")

# ===== Single metrics =====
def eval_vqa(results: dict):
    """BLIP-VQA (attribute binding)."""
    # [FIX] 进入 BLIPvqa_eval 目录运行，这样脚本内部的相对路径才正确
    # out_dir 指向 ../examples
    run_cmd("python BLIP_vqa.py --out_dir=../examples", cwd=ROOT / "BLIPvqa_eval")
    
    res_path = EXAMPLES / "annotation_blip" / "vqa_result.json" # BLIP 输出在这个位置
    
    if res_path.exists():
        results["VQA"] = json.load(open(res_path))
    else:
        results["VQA"] = "No result produced."

def eval_2d_spatial(results: dict):
    """UniDet 2D spatial evaluation."""
    # [FIX] 进入 UniDet_eval 目录运行
    run_cmd("python 2D_spatial_eval.py --outpath=../examples", cwd=ROOT / "UniDet_eval")
    
    r = EXAMPLES / "labels" / "annotation_obj_detection_2d" / "vqa_result.json"
    if r.exists():
        results["2D_Spatial"] = json.load(open(r))
    else:
        results["2D_Spatial"] = "No results."

def eval_numeracy(results: dict):
    """UniDet numeracy evaluation."""
    # [FIX] 进入 UniDet_eval 目录运行
    run_cmd("python numeracy_eval.py --outpath=../examples", cwd=ROOT / "UniDet_eval")
    
    r = EXAMPLES / "annotation_num" / "vqa_result.json"
    if r.exists():
        results["Numeracy"] = json.load(open(r))
    else:
        results["Numeracy"] = "No results."

def eval_3d_spatial(results: dict):
    """UniDet 3D spatial evaluation (depth + detection)."""
    ensure_dirs()
    # [FIX] 进入 UniDet_eval 目录运行
    run_cmd("python 3D_spatial_eval.py --outpath=../examples", cwd=ROOT / "UniDet_eval")
    
    fix_depth_structure()
    r = EXAMPLES / "labels" / "annotation_obj_detection_3d" / "vqa_result.json"
    if r.exists():
        results["3D_Spatial"] = json.load(open(r))
    else:
        results["3D_Spatial"] = "No results."

def eval_clip_similarity(results: dict):
    """CLIPScore for non-spatial relationships."""
    # [FIX] 进入 CLIPScore_eval 目录运行
    run_cmd("python CLIP_similarity.py --outpath=../examples", cwd=ROOT / "CLIPScore_eval")
    
    r = EXAMPLES / "annotation_clip" / "vqa_result.json"
    if r.exists():
        results["CLIP_Similarity"] = json.load(open(r))
    else:
        results["CLIP_Similarity"] = "No results."

# ===== 3-in-1 (official combined metric) =====
def eval_three_in_one(results: dict):
    """Run official 3-in-1 evaluator (attribute + spatial combo)."""
    # [FIX] 进入 3_in_1_eval 目录运行
    # 3_in_1.py 默认读取 data_path=../examples/dataset，这在子目录下是正确的
    run_cmd("python 3_in_1.py --outpath=../examples", cwd=ROOT / "3_in_1_eval")
    
    r = EXAMPLES / "annotation_3_in_1" / "vqa_result.json"
    if r.exists():
        results["3_in_1"] = json.load(open(r))
    else:
        results["3_in_1"] = "No results."

# ===== MLLM (GPT-4V / GPT-4o etc.) =====
GPT4V_CATEGORIES = os.getenv("GPT4V_CATEGORIES", "complex").split(",")
GPT4V_START = int(os.getenv("GPT4V_START", "0"))
GPT4V_STEP = int(os.getenv("GPT4V_STEP", "1"))

def eval_mllm_gpt4v(results: dict):
    """Run GPT-4V / GPT-4o evaluation."""
    mllm_scores = {}
    for cat in GPT4V_CATEGORIES:
        cat = cat.strip()
        if not cat:
            continue
        
        # [FIX] 进入 MLLM_eval 目录运行
        cmd = (
            f'python gpt4v_eval.py '
            f'--category "{cat}" --start {GPT4V_START} --step {GPT4V_STEP}'
        )
        run_cmd(cmd, cwd=ROOT / "MLLM_eval")
        
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
    try:
        eval_vqa(results)
    except Exception as e:
        print(f"⚠️ VQA evaluation failed: {e}")

    try:
        eval_2d_spatial(results)
    except Exception as e:
        print(f"⚠️ 2D Spatial evaluation failed: {e}")

    try:
        eval_numeracy(results)
    except Exception as e:
        print(f"⚠️ Numeracy evaluation failed: {e}")

    try:
        eval_3d_spatial(results)
    except Exception as e:
        print(f"⚠️ 3D Spatial evaluation failed: {e}")

    try:
        eval_clip_similarity(results)
    except Exception as e:
        print(f"⚠️ CLIP evaluation failed: {e}")

    # 2. 官方 3-in-1 综合指标
    try:
        eval_three_in_one(results)
    except Exception as e:
        print(f"⚠️ 3-in-1 evaluation failed: {e}")

    # 3. 多模态大模型评测（GPT-4V / GPT-4o）
    # 默认不跑，除非配置了 API Key
    if os.getenv("OPENAI_API_KEY"):
        try:
            eval_mllm_gpt4v(results)
        except Exception as e:
            print(f"⚠️ MLLM evaluation failed: {e}")
            results["MLLM_GPT4V"] = f"Failed: {e}"
    else:
        print("ℹ️ Skipping MLLM evaluation (OPENAI_API_KEY not set)")

    # 总结到一个 json 里
    summary_path = ROOT / "final_eval_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n\n🎉 ALL DONE! Results saved to:")
    print(summary_path)

if __name__ == "__main__":
    main()