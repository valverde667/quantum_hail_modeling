import pandas as pd
import glob
import os


def collect_results(results_dir="results", output_file="qcbm_summary.csv"):
    """
    Collect all individual QCBM result CSVs into a single dataframe.

    Parameters
    ----------
    results_dir : str
        Path to the directory containing the individual result CSVs.
    output_file : str
        Filename for the aggregated output CSV.

    Returns
    -------
    df_all : pandas.DataFrame
        Combined dataframe of all runs.
    """
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not all_files:
        print("No result CSVs found.")
        return None

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df["filename"] = os.path.basename(f)  # track origin
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values(by="seed").reset_index(drop=True)
    df_all.to_csv(output_file, index=False)
    print(f"Combined {len(df_list)} files into {output_file}")
    return df_all


# Run standalone
if __name__ == "__main__":
    collect_results()
