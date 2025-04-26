import argparse
import subprocess


def go(csv, ref, kl_threshold, min_price, max_price):
    
    cmd = [
        "pytest",
        "tests",
        f"--csv={csv}",
        f"--ref={ref}",
        f"--kl_threshold={kl_threshold}",
        f"--min_price={min_price}",
        f"--max_price={max_price}",
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data quality tests with pytest")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()

    go(
        csv=args.csv,
        ref=args.ref,
        kl_threshold=args.kl_threshold,
        min_price=args.min_price,
        max_price=args.max_price,
    )