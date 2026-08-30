// SPDX-License-Identifier: MIT
//! puretd-engine — PureTD spoken as a Backgammon Engine Protocol engine.
//!
//! The whole engine is this binary plus a weights file. PureTD trains in Python
//! and exports a framework-independent `.bin`; from there the move generator and
//! the forward pass are C, and PyTorch never leaves the build machine. That is
//! deliberate — the desktop packs are downloaded per user, and a bundled Python
//! runtime would be two orders of magnitude larger than the thing it runs.
//!
//! Upstream: <https://github.com/alexstrehl/backgammon-ai-engine> (MIT), forked
//! to customation/backgammon-ai-engine. The networks are self-play trained, so
//! the MIT grant covers the weights as well as the code — they are not distilled
//! from gnubg, whose weights are GPL.

mod board;
mod ffi;
mod model;

use std::io;

use bep_protocol::contract::{error_codes, kinds, methods, Conventions, Describe, EngineIdentity,
    EvaluateParams, Level};
use bep_protocol::gnubg_ids::{decode_match_id, decode_position_id};
use bep_protocol::jsonrpc::{self, codes, FrameSink};
use serde_json::Value;

use model::Engine;

const PROTOCOL_VERSION: &str = "0.1";

/// PureTD evaluates a position with one forward pass and no lookahead, which is
/// what gnubg calls 0-ply and XG calls 1-ply. BEP declares the XG convention, so
/// this is "1ply".
const LEVEL_1PLY: &str = "1ply";

fn describe() -> Describe {
    Describe {
        protocol_version: PROTOCOL_VERSION.to_string(),
        engine: EngineIdentity {
            family: "puretd".to_string(),
            display_name: "PureTD".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            build: concat!("puretd-engine ", env!("CARGO_PKG_VERSION")).to_string(),
        },
        // One model, one scratch buffer, no locking. Concurrency is replicas,
        // exactly as the cloud workers scale.
        max_parallel: 1,
        conventions: Conventions {
            ply_counting: "xg".to_string(),
            equity: "contract".to_string(),
        },
        levels: vec![Level {
            id: LEVEL_1PLY.to_string(),
            kind: kinds::PLY.to_string(),
            display_name: Some("PureTD".to_string()),
            ply_depth: Some(1),
            rollout: None,
            // Named explicitly, and evaluateCube is deliberately absent. PureTD
            // learns doubling as actions inside the RL rather than producing the
            // no-double / take / drop equities the cube contract asks for, and
            // one scalar cannot be taken apart into three. A host is told what
            // this engine answers instead of discovering it from an error.
            methods: Some(vec![
                methods::EVALUATE_POSITION.to_string(),
                methods::EVALUATE_MOVES.to_string(),
                methods::ANALYZE_MOVE.to_string(),
            ]),
            configurable: false,
            supports_progress: false,
            supports_cancel: false,
        }],
    }
}

fn handle(engine: &mut Engine, method: &str, params: &EvaluateParams, id: &Value) -> Value {
    let board = match decode_position_id(&params.position_id) {
        Ok(board) => board,
        Err(e) => return jsonrpc::error(Some(id), error_codes::INVALID_ID, &e.to_string()),
    };
    let context = match decode_match_id(&params.match_id) {
        Ok(context) => context,
        Err(e) => return jsonrpc::error(Some(id), error_codes::INVALID_ID, &e.to_string()),
    };
    if params.level != LEVEL_1PLY {
        return jsonrpc::error(
            Some(id),
            error_codes::UNKNOWN_LEVEL,
            &format!("unknown level {:?}", params.level),
        );
    }

    let result = match method {
        methods::EVALUATE_POSITION => engine
            .evaluate_position(&board)
            .and_then(|e| serde_json::to_value(e).map_err(|e| e.to_string())),
        methods::EVALUATE_CUBE => engine
            .evaluate_cube(&board, &context)
            .and_then(|e| serde_json::to_value(e).map_err(|e| e.to_string())),
        methods::EVALUATE_MOVES | methods::ANALYZE_MOVE => {
            let (Some(die1), Some(die2)) = (params.die1, params.die2) else {
                return jsonrpc::error(Some(id), codes::INVALID_PARAMS, "die1 and die2 are required");
            };
            let (die1, die2) = if die1 <= die2 { (die1, die2) } else { (die2, die1) };
            match engine.evaluate_moves(&board, &params.position_id, &params.match_id, die1, die2) {
                Err(e) => Err(e),
                Ok(moves) => {
                    if method == methods::EVALUATE_MOVES {
                        serde_json::to_value(moves).map_err(|e| e.to_string())
                    } else {
                        match model::analysis_for(moves, params.played_move.as_deref()) {
                            Some(analysis) => {
                                serde_json::to_value(analysis).map_err(|e| e.to_string())
                            }
                            None => Err("no legal play to analyse".to_string()),
                        }
                    }
                }
            }
        }
        _ => unreachable!("dispatch matches evaluation methods only"),
    };

    match result {
        Ok(value) => jsonrpc::success(id, value),
        Err(message) => jsonrpc::error(Some(id), codes::INTERNAL_ERROR, &message),
    }
}

fn main() {
    let mut engine = match Engine::from_env() {
        Ok(engine) => engine,
        Err(message) => {
            // Before any protocol traffic: a host that cannot get weights needs
            // the reason on stderr, not a describe that promises evaluations the
            // engine cannot perform.
            eprintln!("puretd-engine: {message}");
            std::process::exit(1);
        }
    };

    let sink = FrameSink::new(io::stdout());
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let send = |message: &Value| {
        if let Err(e) = sink.send(message) {
            eprintln!("cannot write to stdout ({e}); exiting");
            std::process::exit(1);
        }
    };

    loop {
        let message = match jsonrpc::read_message(&mut reader) {
            Ok(Some(Ok(message))) => message,
            Ok(Some(Err(parse_error))) => {
                send(&jsonrpc::error(None, codes::PARSE_ERROR, &parse_error.to_string()));
                continue;
            }
            Ok(None) => break,
            Err(io_error) => {
                eprintln!("stdin read failed: {io_error}");
                break;
            }
        };

        match message.method.as_str() {
            methods::DESCRIBE => {
                if let Some(id) = &message.id {
                    match serde_json::to_value(describe()) {
                        Ok(result) => send(&jsonrpc::success(id, result)),
                        Err(e) => send(&jsonrpc::error(
                            Some(id),
                            codes::INTERNAL_ERROR,
                            &e.to_string(),
                        )),
                    }
                }
            }
            methods::SHUTDOWN => {
                if let Some(id) = &message.id {
                    send(&jsonrpc::success(id, Value::Null));
                }
                std::process::exit(0);
            }
            // Evaluations are a single forward pass; there is no window in which
            // a cancel could arrive and still mean anything.
            methods::CANCEL => {}
            methods::EVALUATE_POSITION
            | methods::EVALUATE_CUBE
            | methods::EVALUATE_MOVES
            | methods::ANALYZE_MOVE => {
                let Some(id) = message.id else {
                    eprintln!("{} sent as a notification — ignored", message.method);
                    continue;
                };
                match serde_json::from_value::<EvaluateParams>(message.params.unwrap_or(Value::Null))
                {
                    Ok(params) => {
                        let reply = handle(&mut engine, &message.method, &params, &id);
                        send(&reply);
                    }
                    Err(e) => send(&jsonrpc::error(Some(&id), codes::INVALID_PARAMS, &e.to_string())),
                }
            }
            other => {
                if let Some(id) = &message.id {
                    send(&jsonrpc::error(
                        Some(id),
                        codes::METHOD_NOT_FOUND,
                        &format!("unknown method {other:?}"),
                    ));
                }
            }
        }
    }
}
