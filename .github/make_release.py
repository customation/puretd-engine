#!/usr/bin/env python3
"""Turn a built pack into the layout gammonbase.com's engine catalogue reads.

The catalogue is not a document anybody edits. EngineCatalogueService lists the
release bucket under `engines/` and builds it from manifests it finds, so
publishing an engine means putting files in the right places with the right
names:

    engines/<id>/identity.json                     who the engine is
    engines/<id>/<version>/<platform>/release.json  one published build
    engines/<id>/<version>/<platform>/<archive>     the zip the app downloads

This writes that tree. Uploading it is a separate step needing bucket
credentials; producing it is not, so CI produces it on every build and the
upload becomes a copy rather than a construction.

`platform` uses the tokens the client enumerates in current_platform() —
windows-x86_64, macos-aarch64, macos-x86_64, linux-x86_64, linux-aarch64. The
client filters on them and never guesses from a filename, so an unknown token
means a build nobody can install.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

# Rust target triples to the tokens the desktop client knows. Anything absent
# is a platform the app cannot install, and saying so beats publishing a build
# that silently matches nothing.
PLATFORMS = {
    "x86_64-pc-windows-msvc": "windows-x86_64",
    "x86_64-unknown-linux-gnu": "linux-x86_64",
    "aarch64-unknown-linux-gnu": "linux-aarch64",
    "aarch64-apple-darwin": "macos-aarch64",
    "x86_64-apple-darwin": "macos-x86_64",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, help="directory holding the built engine files")
    p.add_argument("--out", required=True, help="where to write the engines/ tree")
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--licence", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--target", required=True, help="Rust target triple")
    p.add_argument("--command", required=True, help="executable name inside the archive")
    p.add_argument("--instances", type=int, default=1)
    args = p.parse_args()

    platform = PLATFORMS.get(args.target)
    if platform is None:
        print(
            f"{args.target} has no client platform token; the app could not install "
            f"this build. Known: {', '.join(sorted(PLATFORMS))}",
            file=sys.stderr,
        )
        return 2

    pack = pathlib.Path(args.pack)
    if not (pack / args.command).exists():
        print(f"{args.command} not found in {pack}", file=sys.stderr)
        return 2

    root = pathlib.Path(args.out) / "engines" / args.id
    release_dir = root / args.version / platform
    release_dir.mkdir(parents=True, exist_ok=True)

    # Identity is per engine, not per build: every platform writes the same
    # bytes, so the last upload wins and they all agree.
    (root / "identity.json").write_text(
        json.dumps(
            {
                "id": args.id,
                "name": args.name,
                "description": args.description,
                "licence": args.licence,
                "sourceUrl": args.source_url,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    archive_stem = f"{args.id}-{args.version}-{platform}"
    archive = pathlib.Path(
        shutil.make_archive(str(release_dir / archive_stem), "zip", root_dir=pack)
    )

    (release_dir / "release.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "platform": platform,
                # A name, not a path: the server composes the URL, so a release
                # cannot point the client somewhere else.
                "file": archive.name,
                "sha256": sha256(archive),
                "sizeBytes": archive.stat().st_size,
                "command": args.command,
                "args": [],
                "instances": args.instances,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{args.id} {args.version} {platform}: {archive.name} "
          f"({archive.stat().st_size / 1048576:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
