//! Loading PureTD's weights and asking them questions.
//!
//! Two things are deliberately NOT done here.
//!
//! No cube evaluation. PureTD does have a cubeful model, but it learns doubling
//! as actions inside the RL rather than deriving no-double / take / drop equities
//! the way BEP's cube contract asks for, and a single scalar cannot be pulled
//! apart into those three numbers. Inventing them would produce confident cube
//! advice the engine never gave, so `describe` declares the methods this level
//! actually answers and `evaluateCube` is simply not one of them.
//!
//! No lookahead. One forward pass per candidate, which is gnubg's 0-ply and XG's
//! 1-ply. PureTD ships an n-ply search in `c_engine/bg_nply.c`; wiring it up is
//! its own piece of work with its own parity gate, not something to slip in.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use bep_protocol::contract::{MoveAnalysis, MoveHint, MovesEvaluation, PositionEvaluation};
use bep_protocol::gnubg_ids::{position_id_storage_base64, Board, MatchContext};

use crate::board::to_puretd;
use crate::ffi::{self, BoardState, Play, MAX_INPUT_SIZE, MAX_PLAYS, NN_PROB5_OUTPUTS, OFF_SENTINEL,
    BAR_SENTINEL};

/// Where the weights are, when nothing says otherwise: beside the executable,
/// which is how the installed pack lays an engine out.
const DEFAULT_WEIGHTS: &str = "puretd.bin";
const WEIGHTS_ENV: &str = "PURETD_WEIGHTS";

/// `nn_forward_prob5` returns these five, already sigmoid'd and clamped so the
/// nested events cannot contradict each other.
struct Probabilities {
    win: f64,
    win_gammon: f64,
    win_backgammon: f64,
    lose_gammon: f64,
    lose_backgammon: f64,
}

impl Probabilities {
    /// Cubeless money equity, the same reduction `nn_eval.c` performs.
    fn equity(&self) -> f64 {
        2.0 * self.win + self.win_gammon + self.win_backgammon
            - self.lose_gammon
            - self.lose_backgammon
            - 1.0
    }
}

pub struct Engine {
    model: ffi::NNModel,
    input: Vec<f32>,
}

impl Engine {
    pub fn from_env() -> Result<Self, String> {
        let path = match std::env::var(WEIGHTS_ENV) {
            Ok(value) if !value.is_empty() => PathBuf::from(value),
            _ => beside_executable(DEFAULT_WEIGHTS)?,
        };
        Self::load(&path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!(
                "weights not found at {}. Point {WEIGHTS_ENV} at a .bin exported by \
                 export_weights.py, or place {DEFAULT_WEIGHTS} beside the executable.",
                path.display()
            ));
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| format!("weights path {} is not usable as a C string", path.display()))?;

        let mut model = ffi::NNModel::default();
        // SAFETY: `model` is owned and correctly shaped; `c_path` is a valid
        // NUL-terminated string that outlives the call.
        let rc = unsafe { ffi::nn_load(&mut model, c_path.as_ptr()) };
        if rc != 0 {
            return Err(format!("{} is not a BGNN model file", path.display()));
        }
        if model.output_mode != 2 {
            // SAFETY: nn_load succeeded, so the model owns allocations to release.
            unsafe { ffi::nn_free(&mut model) };
            return Err(format!(
                "{} is output_mode {} — this engine needs a prob5 model, because the \
                 protocol reports win/gammon/backgammon probabilities and a single \
                 equity scalar cannot be taken apart into them.",
                path.display(),
                model.output_mode
            ));
        }
        let input_size = model.input_size as usize;
        if input_size > MAX_INPUT_SIZE {
            unsafe { ffi::nn_free(&mut model) };
            return Err(format!("model wants {input_size} inputs, more than expected"));
        }

        Ok(Self { model, input: vec![0.0; input_size] })
    }

    fn probabilities(&mut self, state: &BoardState) -> Probabilities {
        // SAFETY: the buffer is at least the model's input size, checked at load.
        unsafe { ffi::encode_state(state, self.input.as_mut_ptr()) };
        self.forward()
    }

    fn forward(&mut self) -> Probabilities {
        let mut probs = [0.0f32; NN_PROB5_OUTPUTS];
        // SAFETY: model is loaded, input is input_size long, probs is 5 long.
        unsafe {
            ffi::nn_forward_prob5(&self.model, self.input.as_ptr(), probs.as_mut_ptr());
        }
        Probabilities {
            win: probs[0] as f64,
            win_gammon: probs[1] as f64,
            win_backgammon: probs[2] as f64,
            lose_gammon: probs[3] as f64,
            lose_backgammon: probs[4] as f64,
        }
    }

    pub fn evaluate_position(&mut self, board: &Board) -> Result<PositionEvaluation, String> {
        let state = to_puretd(board);
        let p = self.probabilities(&state);
        Ok(PositionEvaluation {
            equity: p.equity(),
            // Cubeless: this level owns no cube model, and repeating the
            // cubeless number under a cubeful name would be a claim, not a value.
            cubeful_equity: p.equity(),
            win_prob: p.win,
            win_gammon: p.win_gammon,
            win_backgammon: p.win_backgammon,
            lose_gammon: p.lose_gammon,
            lose_backgammon: p.lose_backgammon,
        })
    }

    pub fn evaluate_cube(
        &mut self,
        _board: &Board,
        _context: &MatchContext,
    ) -> Result<serde_json::Value, String> {
        Err("this engine does not answer cube decisions; describe lists the methods \
             this level supports"
            .to_string())
    }

    pub fn evaluate_moves(
        &mut self,
        board: &Board,
        position_id: &str,
        match_id: &str,
        die1: i32,
        die2: i32,
    ) -> Result<MovesEvaluation, String> {
        if !(1..=6).contains(&die1) || !(1..=6).contains(&die2) {
            return Err(format!("dice {die1}-{die2} are not both 1..6"));
        }
        let state = to_puretd(board);

        let mut plays = vec![Play::default(); MAX_PLAYS];
        let mut features = vec![0.0f32; MAX_PLAYS * self.input.len()];
        // SAFETY: both buffers are MAX_PLAYS-sized as the C requires, and the
        // feature buffer is plays * input_size as documented in bg_engine.h.
        let count = unsafe {
            ffi::get_legal_plays_encoded(
                &state,
                die1,
                die2,
                plays.as_mut_ptr(),
                MAX_PLAYS as i32,
                features.as_mut_ptr(),
            )
        };
        if count < 0 {
            return Err("move generation overflowed".to_string());
        }

        let stride = self.input.len();
        let mut scored: Vec<(f64, Probabilities, usize)> = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            self.input.copy_from_slice(&features[index * stride..(index + 1) * stride]);
            let p = self.forward();
            // The encoded features describe the position AFTER the play, with
            // the opponent on roll, so the network's answer is from their side.
            // Negating puts every candidate back in the mover's terms, which is
            // what "best play" has to be ranked in.
            let equity = -p.equity();
            scored.push((equity, p, index));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let best_equity = scored.first().map(|s| s.0).unwrap_or(0.0);
        let storage_id = position_id_storage_base64(position_id).unwrap_or_default();
        let alternatives = scored
            .iter()
            .enumerate()
            .map(|(rank, (equity, p, index))| MoveHint {
                gnubg_position_id: storage_id.clone(),
                gnubg_match_id: match_id.to_string(),
                die1,
                die2,
                evaluation_engine_id: 0,
                plies: 1,
                rank: rank as i32 + 1,
                move_notation: notation(&plays[*index]),
                equity: *equity,
                error_vs_best: best_equity - *equity,
                // Reported from the MOVER's side, so the win/lose halves swap
                // with the equity above.
                win_prob: 1.0 - p.win,
                win_gammon: p.lose_gammon,
                win_backgammon: p.lose_backgammon,
                lose_gammon: p.win_gammon,
                lose_backgammon: p.win_backgammon,
                evaluated_utc: now_utc(),
            })
            .collect();

        Ok(MovesEvaluation { die1, die2, alternatives })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: the model was loaded by nn_load and is freed exactly once.
        unsafe { ffi::nn_free(&mut self.model) };
    }
}

/// Pick out the played alternative, falling back to the best when the host did
/// not name one (or named one this engine did not generate).
pub fn analysis_for(moves: MovesEvaluation, played: Option<&str>) -> Option<MoveAnalysis> {
    let best = moves.alternatives.first()?.clone();
    let played = played
        .and_then(|notation| moves.alternatives.iter().find(|h| h.move_notation == notation))
        .cloned()
        .unwrap_or_else(|| best.clone());
    Some(MoveAnalysis { played, best })
}

/// Standard notation for one play: `8/5 6/5`, `bar/20`, `6/off`.
fn notation(play: &Play) -> String {
    let point = |slot: i32| -> String {
        match slot {
            BAR_SENTINEL => "bar".to_string(),
            OFF_SENTINEL => "off".to_string(),
            // PureTD indexes 0..23; players count points 1..24.
            index => (index + 1).to_string(),
        }
    };
    (0..play.num_moves as usize)
        .map(|i| format!("{}/{}", point(play.moves[i].src), point(play.moves[i].dst)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The contract carries an evaluation timestamp; seconds are plenty and this
/// avoids a chrono dependency for one field.
fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let rem = secs % 86_400;
    // Civil-from-days, Howard Hinnant's algorithm.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn beside_executable(name: &str) -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this executable: {e}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| "this executable has no parent directory".to_string())?;
    Ok(dir.join(name))
}
