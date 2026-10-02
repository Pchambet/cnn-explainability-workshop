"""Command line: ``cnn-xai data | run [--only ...] | report``."""

from __future__ import annotations

import argparse

EXPERIMENTS = ("portrait", "lfw", "latency")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cnn-xai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="download LFW (idempotent, ~230 MB)")
    run = sub.add_parser("run", help="run the experiments and write results/")
    run.add_argument("--only", choices=EXPERIMENTS, action="append", help="repeatable")
    run.add_argument("--threads", type=int, default=3, help="TensorFlow CPU threads")
    sub.add_parser("report", help="figures in docs/figures/ and the HTML report in site/")
    args = parser.parse_args(argv)

    if args.command == "data":
        from cnn_explainability import lfw

        print(lfw.download())
    elif args.command == "run":
        from cnn_explainability import pipeline, vgg

        vgg.limit_threads(args.threads)
        for name in args.only or EXPERIMENTS:
            print(f"[{name}]")
            if name == "latency":
                pipeline.run_latency(threads=args.threads)
            else:
                getattr(pipeline, f"run_{name}")()
    else:
        from cnn_explainability import report

        report.build()


if __name__ == "__main__":
    main()
