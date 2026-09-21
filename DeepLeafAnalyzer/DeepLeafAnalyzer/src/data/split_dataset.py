import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df, seed=42):
    train, temp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['class_id'])
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp['class_id'])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    tr, va, te = stratified_split(df, args.seed)
    tr.to_csv(out/'train.csv', index=False); va.to_csv(out/'val.csv', index=False); te.to_csv(out/'test.csv', index=False)
    print(f'train={len(tr)} val={len(va)} test={len(te)}')

if __name__ == '__main__':
    main()
