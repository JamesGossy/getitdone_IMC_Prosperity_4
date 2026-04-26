import itertools
import re
import subprocess
import tempfile
import os
import time

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
TRADING_FILE = "trader.py"
ROUNDS = ["3"]

# Round 3 parameter grid — sweep options IV scalping + VEV mean-reversion thresholds
PARAM_GRID = {
    "THR_OPEN":        [0.3, 0.5, 0.7],    # open position when |deviation| exceeds this
    "IV_SCALPING_THR": [0.5, 0.7, 1.0],    # switch_mean threshold to activate scalping
    "VEV_MR_THR":      [10, 15, 20],        # VEV mean-reversion threshold in ticks
}

# Maps parameter names to their Regex search pattern and replacement string.
REGEX_MAP = {
    "THR_OPEN":        (r"^THR_OPEN\s*=\s*[\d\.]+", "THR_OPEN         = {value}"),
    "IV_SCALPING_THR": (r"^IV_SCALPING_THR\s*=\s*[\d\.]+", "IV_SCALPING_THR  = {value}"),
    "VEV_MR_THR":      (r"^VEV_MR_THR\s*=\s*[\d\.]+", "VEV_MR_THR    = {value}"),
}
# ────────────────────────────────────────────────────────────────────────────

def run_backtest(**params):
    """
    Reads the original file, dynamically swaps magic numbers using regex, 
    runs the backtester, and returns the total PnL.
    """
    with open(TRADING_FILE, "r", encoding='utf-8') as f:
        code = f.read()
    
    # 1. Hot-swap the parameters using our Regex Map
    for param_name, value in params.items():
        if param_name in REGEX_MAP:
            pattern, template = REGEX_MAP[param_name]
            code = re.sub(pattern, template.format(value=value), code, flags=re.MULTILINE)
    
    # 2. Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code)
        temp_file_name = temp_file.name
        
    try:
        # 3. Trigger the backtester CLI for each round and sum PnLs
        pnl = 0.0
        for round_num in ROUNDS:
            cmd = ["prosperity4btest", temp_file_name, round_num, "--data", "./data"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            pnl += extract_pnl(result.stdout)
        
    finally:
        # 5. Clean up
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
            
    return pnl

def extract_pnl(stdout):
    """Parses the CLI output table to find the Total PnL."""
    clean_stdout = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', stdout)
    
    lines = clean_stdout.strip().split('\n')
    for line in reversed(lines):
        if "| Total" in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            try:
                return float(parts[2].replace(',', ''))
            except (ValueError, IndexError):
                continue
                
    matches = re.findall(r'Total profit:\s*([-+]?\d+(?:,\d+)*(?:\.\d+)?)', clean_stdout)
    if matches:
        return float(matches[-1].replace(',', ''))
        
    return 0.0

if __name__ == "__main__":
    results = []
    
    # Setup dynamic grid combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    
    print(f"🚀 Starting sweep of {len(combinations)} parameter combinations...\n")
    start_time = time.time()
    
    for i, combo in enumerate(combinations, 1):
        # Pack the current combination into a dictionary
        current_params = dict(zip(keys, combo))
        
        # Format terminal output dynamically
        param_str = ", ".join(f"{k}={v}" for k, v in current_params.items())
        print(f"[{i}/{len(combinations)}] Testing {param_str}...")
        
        # Run backtest with unpacked kwargs
        pnl = run_backtest(**current_params)
        print(f"      ↳ PnL: {pnl:,.2f}")
        
        # Save results
        current_params["PnL"] = pnl
        results.append(current_params)
        
    # Sort results by highest PnL
    results.sort(key=lambda x: x["PnL"], reverse=True)
    elapsed = time.time() - start_time
    
    print("\n" + "═"*90)
    print("🏆 TOP 5 PARAMETER COMBINATIONS 🏆")
    print("═"*90)
    for i, res in enumerate(results[:5], 1):
        param_str = " | ".join(f"{k}: {res[k]}" for k in keys)
        print(f" {i}. PnL: {res['PnL']:>10,.2f}  |  {param_str}")
    print("═"*90)
    print(f"Sweep completed in {elapsed:.1f} seconds.")