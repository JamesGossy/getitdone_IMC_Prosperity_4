import itertools
import re
import subprocess
import tempfile
import os
import time

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
TRADING_FILE = "trader.py"
ROUNDS = ["1", "2"]

# Define the ranges you want to test. 
PARAM_GRID = {
    "ACO_FAIR": [10000],
    "IPR_SLOPE": [0.001],
    "ACO_SKEW": [2],                        # locked
    "IPR_BUY_THRESH": [3, 4, 5],            # zoom around optimal
    "IPR_SELL_THRESH": [6],                 # dead code at tight ask, just pick one
    "IPR_PASSIVE_ASK": [3,4,5],        # push tighter than sweep 1
    "IPR_PASSIVE_BID": [0, 1, 2]            # NEW — baseline is 1 (floor-1)
}

# Maps parameter names to their Regex search pattern and replacement string.
# This safely edits the hardcoded numbers deep inside your logic 
# without touching the original file.
REGEX_MAP = {
    "ACO_FAIR": (r"^ACO_FAIR\s*=\s*[\d_]+", "ACO_FAIR = {value}"),
    "IPR_SLOPE": (r"^IPR_SLOPE\s*=\s*[\d\.]+", "IPR_SLOPE = {value}"),
    "ACO_SKEW": (r"int\(position\s*/\s*limit\s*\*\s*[\d\.]+\)", "int(position / limit * {value})"),
    "IPR_PASSIVE_BID": (r"math\.floor\(fair\)\s*-\s*[\d\.]+", "math.floor(fair) - {value}"),
    "IPR_BUY_THRESH": (r"ask\s*>\s*fair\s*\+\s*[\d\.]+", "ask > fair + {value}"),
    "IPR_SELL_THRESH": (r"bid\s*<\s*fair\s*\+\s*[\d\.]+", "bid < fair + {value}"),
    "IPR_PASSIVE_ASK": (r"math\.ceil\(fair\)\s*\+\s*[\d\.]+", "math.ceil(fair) + {value}")\
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