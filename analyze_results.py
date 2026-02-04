import pandas as pd
import argparse
import os
import yaml
import re
from rich.console import Console
from rich.table import Table

def main(args):
    console = Console()
    
    # 1. Load Test Results
    res_path = os.path.join(args.exp_dir, "test_metrics.csv")
    if not os.path.exists(res_path):
        console.print(f"[red]Error: Results not found at {res_path}[/]")
        return
    df_res = pd.read_csv(res_path)
    
    # 2. Load Metadata (to get overlap_ratio)
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    test_dir = None
    if os.path.exists(conf_path):
        with open(conf_path) as f:
            conf = yaml.safe_load(f)
        test_dir = conf["data"].get("valid_dir")
    
    if not test_dir or not os.path.exists(test_dir):
        # Fallback based on typical structure or previously seen paths
        test_dir = "/nfs/data/speech-data/nm/nm_v15/test"
        console.print(f"[yellow]Warning: Could not determine valid test_dir from config. Using fallback: {test_dir}[/]")

    # Detect version from test_dir (e.g., nm_v15 -> v15)
    match = re.search(r"nm_(v\d+)", test_dir)
    version = match.group(1) if match else "v15"
    split = "test" if "test" in test_dir else "train"
    
    meta_path = os.path.join(test_dir, f"nm_{version}_{split}_2sp.csv")
    
    if not os.path.exists(meta_path):
        # Try to find any 2sp.csv file in the directory
        cands = [f for f in os.listdir(test_dir) if "2sp.csv" in f]
        if cands:
            meta_path = os.path.join(test_dir, cands[0])
            console.print(f"[yellow]Exact metadata match not found. Using best candidate: {meta_path}[/]")
        else:
            console.print(f"[red]Error: Metadata not found at {test_dir}[/]")
            return
        
    console.print(f"[blue]Loading metadata from {meta_path}...[/]")
    df_meta = pd.read_csv(meta_path)
    
    # 3. Merge Results and Metadata
    if len(df_res) != len(df_meta):
        console.print(f"[yellow]Warning: Row count mismatch! Results: {len(df_res)}, Metadata: {len(df_meta)}[/]")
        min_len = min(len(df_res), len(df_meta))
        df_res = df_res.iloc[:min_len].reset_index(drop=True)
        df_meta = df_meta.iloc[:min_len].reset_index(drop=True)
    
    # Assign overlap_ratio to results
    if "overlap_ratio" in df_meta.columns:
        df_res["overlap_ratio"] = df_meta["overlap_ratio"].round(1)
    else:
        console.print("[red]Error: 'overlap_ratio' column missing in metadata.[/]")
        return
    
    # 4. Group by Overlap Ratio
    group_col = "overlap_ratio"
    grouped = df_res.groupby(group_col)[["si_sdr", "si_sdri", "stoi", "pesq", "wer"]].mean()
    count_grouped = df_res.groupby(group_col)["si_sdr"].count()
    
    # 5. Print Table
    table = Table(title="LLM-TSE Performance Analysis by Overlap Ratio")
    table.add_column("Overlap", style="cyan", justify="center")
    table.add_column("Count", style="white", justify="right")
    table.add_column("SI-SDR (dB)", style="magenta", justify="right")
    table.add_column("SI-SDRi (dB)", style="bright_magenta", justify="right")
    table.add_column("STOI", style="green", justify="right")
    table.add_column("PESQ", style="blue", justify="right")
    table.add_column("WER", style="yellow", justify="right")
    
    total_metrics = df_res[["si_sdr", "si_sdri", "stoi", "pesq", "wer"]].mean()
    total_row = pd.DataFrame(total_metrics).T
    total_row.index = ["Total"]
    
    table.add_row(
        "Total",
        str(len(df_res)),
        f"{total_metrics['si_sdr']:.2f}",
        f"{total_metrics['si_sdri']:.2f}",
        f"{total_metrics['stoi']:.3f}",
        f"{total_metrics['pesq']:.3f}",
        f"{total_metrics['wer']:.3f}",
        style="bold"
    )
    table.add_section()
    
    for ratio, row in grouped.iterrows():
        count = count_grouped[ratio]
        table.add_row(
            f"{ratio:.1f}",
            str(count),
            f"{row['si_sdr']:.2f}",
            f"{row['si_sdri']:.2f}",
            f"{row['stoi']:.3f}",
            f"{row['pesq']:.3f}",
            f"{row['wer']:.3f}"
        )
        
    console.print(table)
    
    # Save analysis
    final_df = pd.concat([grouped, total_row])
    final_df["count"] = list(count_grouped) + [len(df_res)]
    
    out_path = os.path.join(args.exp_dir, "analysis_by_overlap.csv")
    final_df.round(3).to_csv(out_path)
    console.print(f"\n[blue]Analysis saved to {out_path}[/]")
    
    full_path = os.path.join(args.exp_dir, "test_metrics_full.csv")
    df_res.to_csv(full_path, index=False)
    console.print(f"[blue]Full metrics with overlap info saved to {full_path}[/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
