import argparse
from typing import Any


def solver_settings(
    B, X=6, Bbase=16, Ibase=90e3, s1base=60e3, s2base=20e3, lrbase=0.02
):
    R = B / 16
    I = int(Ibase * (X / R))
    s1 = int(I - s1base / R)
    s2 = int(I - s2base / R)
    lr = lrbase * R
    return {
        "IMS_PER_BATCH": B,
        "MAX_ITER": I,
        "STEPS": (s1, s2),
        "BASE_LR": lr,
    }


def _validate_positive_int(value: Any) -> int:
    val = int(value)
    if not val > 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {val}")
    return val


def _validate_positive_float(value: Any) -> float:
    val = float(value)
    if not val > 0:
        raise argparse.ArgumentTypeError(f"expected positive number, got {val}")
    return val


def get_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-b",
        "--batch-size",
        type=_validate_positive_int,
        default=16,
        help="batch size. (default: %(default)s)",
    )
    ap.add_argument(
        "-x",
        "--x-schedule",
        type=_validate_positive_float,
        default=6,
        help='solver "x" schedule. (default: %(default)s)',
    )
    ap.add_argument("--_lr-base", type=_validate_positive_float, default=0.02)
    return ap


if __name__ == "__main__":
    args: argparse.Namespace = get_argparser().parse_args()
    result = solver_settings(B=args.batch_size, X=args.x_schedule, lrbase=args._lr_base)
    print(result)
