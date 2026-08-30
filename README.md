# puretd-engine

PureTD spoken as a [Backgammon Engine Protocol](https://github.com/customation/bep)
engine: JSON-RPC 2.0 over stdin/stdout, no Python at runtime.

The whole engine is one binary and one weights file — around **2.4 MB**
together. PureTD trains in Python and exports a framework-independent `.bin`;
from there the move generator and the network forward pass are C, and PyTorch
never leaves the build machine. That matters because the desktop packs are
downloaded per user: a bundled Python runtime with PyTorch would be two orders
of magnitude larger than the thing it runs.

## Provenance and licence

The engine is Alexander Strehl's [PureTD](https://github.com/alexstrehl/backgammon-ai-engine),
MIT licensed, forked to [customation/backgammon-ai-engine](https://github.com/customation/backgammon-ai-engine).
Its C sources are compiled in unmodified.

The MIT grant covers the networks as well as the code. That is worth stating
explicitly: the weights are trained by self-play, not distilled from GNU
Backgammon, whose own weights are GPL. `requirements.txt` upstream lists
`gnubg-nn` and `bgsage` — those are benchmark opponents under a "Reproduction
environment" comment, not inference dependencies, and neither is needed to
build or run this.

This wrapper is MIT too.

## What it answers

`describe` reports one level, `1ply` — a single forward pass with no lookahead,
which is what gnubg calls 0-ply and XG calls 1-ply. BEP declares the XG
convention.

The level names the methods it supports, and **`evaluateCube` is not among
them**. PureTD does have a cubeful model, but it learns doubling as actions
inside the reinforcement learner rather than producing the no-double / take /
drop equities the cube contract asks for, and a single scalar cannot be taken
apart into three. Answering with invented numbers would be worse than not
answering, so a host is told up front instead of discovering it from an error.

A **prob5** model is required. The flagship cubeful network emits one equity
scalar, which cannot populate the protocol's win/gammon/backgammon fields; the
engine refuses an equity-mode file with an explanation rather than reporting
zeros.

## Building

Needs the fork checked out beside this repository (or `PURETD_SRC` pointing at
it), a C compiler, and a Rust toolchain.

```
git clone https://github.com/customation/backgammon-ai-engine
git clone https://github.com/customation/puretd-engine
cd puretd-engine
cargo build --release
```

Then export a weights file and put it beside the binary as `puretd.bin`, or
point `PURETD_WEIGHTS` at one:

```
python ../backgammon-ai-engine/export_weights.py \
    ../backgammon-ai-engine/best_models/cubeless_prob5_512_512_256_128.pt \
    target/release/puretd.bin
```

## Correctness

Two things are asserted rather than assumed, because both failure modes are
silent — a wrong board or a wrong network produces confident evaluations of a
game nobody is playing.

- **The forward pass matches PyTorch.** `parity_check` in the fork runs the same
  input through both and fails outside 1e-4. Measured: 62.18351364 against
  62.18350601, a difference of 7.6e-06.
- **The board conversion matches PureTD's own.** `cargo test` decodes the
  opening Position ID and asserts the result is point-for-point identical to
  what `board_init` builds. A mirrored or shifted board is a legal position, so
  nothing else would catch it.

As a end-to-end check, the engine's best opening 3-1 is `8/5 6/5` — making the
five point, the most agreed-upon play in the game.
