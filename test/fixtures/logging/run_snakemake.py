"""Run a real dry-run with explicitly simulated external failures for log inspection.

This is a test/demo launcher, never imported by the production workflow. Only
UniProt length requests and shard-registry I/O are intercepted. No jobs run.
"""

import argparse
import io
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", choices=(
        "none", "uniprot-failures", "progress", "registry-corrupt",
        "registry-read-denied", "registry-write-denied",
    ), default="none")
    options, arguments = parser.parse_known_args()
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if "--dry-run" not in arguments:
        parser.error("This launcher requires --dry-run; it must never execute jobs.")
    print(f"[logging-demo] Simulation: {options.simulate}; dry-run only.", flush=True)
    original_urlopen = urllib.request.urlopen
    original_read = Path.read_text
    original_replace = os.replace
    delayed = False

    def urlopen(url, *args, **kwargs):
        nonlocal delayed
        if not str(url).startswith("https://rest.uniprot.org/uniprotkb/"):
            return original_urlopen(url, *args, **kwargs)
        if options.simulate == "progress":
            if not delayed:
                print("[logging-demo] Simulating one slow lookup (31 seconds).", flush=True)
                time.sleep(31)
                delayed = True
            return io.BytesIO(b">simulated\nACDE\n")
        name = str(url).rsplit("/", 1)[-1].removesuffix(".fasta")
        if name == "P12345":
            raise TimeoutError("simulated timeout")
        if name == "Q9Y6K9":
            raise urllib.error.HTTPError(str(url), 404, "simulated missing accession", {}, None)
        if name == "Q68DK7":
            raise urllib.error.HTTPError(str(url), 503, "simulated server outage", {}, None)
        if name == "Q8NEP3":
            raise urllib.error.URLError("simulated connection failure")
        return io.BytesIO(b">simulated\nACDE\n")

    def read_text(path, *args, **kwargs):
        if path.name == ".shard_registry.json":
            if options.simulate == "registry-corrupt":
                return "{simulated invalid json"
            if options.simulate == "registry-read-denied":
                raise PermissionError("simulated registry read denial")
        return original_read(path, *args, **kwargs)

    def replace(source, target, *args, **kwargs):
        if Path(target).name == ".shard_registry.json":
            raise PermissionError("simulated registry write denial")
        return original_replace(source, target, *args, **kwargs)

    if options.simulate in ("uniprot-failures", "progress"):
        urllib.request.urlopen = urlopen
    if options.simulate in ("registry-corrupt", "registry-read-denied"):
        Path.read_text = read_text
    if options.simulate == "registry-write-denied":
        os.replace = replace
    try:
        from snakemake.cli import main as snakemake_main
        return snakemake_main(arguments)
    finally:
        urllib.request.urlopen = original_urlopen
        Path.read_text = original_read
        os.replace = original_replace


if __name__ == "__main__":
    sys.exit(main())
