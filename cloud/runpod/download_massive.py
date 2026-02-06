#!/usr/bin/env python3
"""Descarga masiva de datos de código para entrenar 3B."""

from datasets import load_dataset
import json
from tqdm import tqdm
import os

OUTPUT_DIR = "data/distillation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_github_python(max_samples=1_000_000):
    """Descarga código Python de GitHub."""
    print("🚀 Descargando codeparrot/github-code (Python)...")
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])
    
    count = 0
    output_file = f"{OUTPUT_DIR}/github_python.jsonl"
    
    with open(output_file, "w") as f:
        for item in tqdm(ds, total=max_samples, desc="GitHub Python"):
            code = item.get("code", "")[:3000]
            if 100 < len(code) < 2500:
                f.write(json.dumps({
                    "instruction": f"# {item.get('path', 'code.py')}\nComplete this code:",
                    "output": code
                }) + "\n")
                count += 1
            
            if count >= max_samples:
                break
            
            if count % 100000 == 0 and count > 0:
                print(f"   {count:,} samples...")
    
    print(f"✅ GitHub Python: {count:,} samples")
    return count


def download_evol_instruct(max_samples=100_000):
    """Descarga EvolInstruct Code."""
    print("🚀 Descargando WizardLM/WizardLM_evol_instruct_V2_196k...")
    try:
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
        
        count = 0
        output_file = f"{OUTPUT_DIR}/evol_instruct.jsonl"
        
        with open(output_file, "w") as f:
            for item in tqdm(ds, desc="EvolInstruct"):
                # Filtrar solo los relacionados con código
                instruction = item.get("instruction", "")
                output = item.get("output", "")
                
                code_keywords = ["def ", "class ", "import ", "function", "code", "python", "program"]
                if any(kw in instruction.lower() or kw in output.lower() for kw in code_keywords):
                    f.write(json.dumps({
                        "instruction": instruction[:1000],
                        "output": output[:3000]
                    }) + "\n")
                    count += 1
                
                if count >= max_samples:
                    break
        
        print(f"✅ EvolInstruct Code: {count:,} samples")
        return count
    except Exception as e:
        print(f"⚠️ EvolInstruct falló: {e}")
        return 0


def download_python_code_instructions(max_samples=100_000):
    """Descarga instrucciones de código Python."""
    print("🚀 Descargando iamtarun/python_code_instructions_18k_alpaca...")
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        
        count = 0
        output_file = f"{OUTPUT_DIR}/python_instructions.jsonl"
        
        with open(output_file, "w") as f:
            for item in tqdm(ds, desc="Python Instructions"):
                f.write(json.dumps({
                    "instruction": item.get("instruction", "")[:1000],
                    "output": item.get("output", "")[:3000]
                }) + "\n")
                count += 1
                
                if count >= max_samples:
                    break
        
        print(f"✅ Python Instructions: {count:,} samples")
        return count
    except Exception as e:
        print(f"⚠️ Python Instructions falló: {e}")
        return 0


def main():
    print("=" * 60)
    print("📦 DESCARGA MASIVA DE DATOS DE CÓDIGO")
    print("=" * 60)
    
    total = 0
    
    # 1. GitHub Python (1M)
    total += download_github_python(1_000_000)
    
    # 2. Python Instructions (18K)
    total += download_python_code_instructions(100_000)
    
    # 3. EvolInstruct código
    total += download_evol_instruct(100_000)
    
    print("\n" + "=" * 60)
    print(f"📊 TOTAL: {total:,} samples")
    print(f"📊 Tokens estimados: ~{total * 500 / 1e9:.1f}B")
    print("=" * 60)
    
    # Listar archivos
    print("\n📁 Archivos generados:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".jsonl"):
            size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024 / 1024
            print(f"   {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()
