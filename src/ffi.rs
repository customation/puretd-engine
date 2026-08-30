//! The C surface, declared once.
//!
//! Mirrors `c_engine/bg_engine.h` and `c_inference/nn_eval.h` from the PureTD
//! fork. Every one of these declarations has to match the header exactly — a
//! wrong field order here does not crash, it silently evaluates the wrong
//! board, which is the failure mode this whole engine exists to avoid. The
//! layout is asserted against the C at startup rather than trusted.

use std::os::raw::{c_char, c_float, c_int};

pub const NUM_POINTS: usize = 24;
pub const NUM_FEATURES: usize = 196;
pub const MAX_MOVES_PER_PLAY: usize = 4;
pub const MAX_PLAYS: usize = 4096;
pub const NN_MAX_LAYERS: usize = 8;
pub const NN_PROB5_OUTPUTS: usize = 5;

/// `bg_engine.h`: WHITE is the positive side.
pub const WHITE: c_int = 0;
pub const BLACK: c_int = 1;

pub const BAR_SENTINEL: c_int = -1;
pub const OFF_SENTINEL: c_int = -2;

/// Feature vectors are 196 wide by default, but the flagship networks take 200
/// inputs — the extra columns are match context the encoder appends. The model
/// file states its own input size and that is what the buffers are sized from.
pub const MAX_INPUT_SIZE: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BoardState {
    /// `>0` = WHITE checkers on the point, `<0` = BLACK.
    pub points: [c_int; NUM_POINTS],
    /// Indexed by `WHITE` / `BLACK`.
    pub bar: [c_int; 2],
    pub off: [c_int; 2],
    pub turn: c_int,
}

impl Default for BoardState {
    fn default() -> Self {
        Self { points: [0; NUM_POINTS], bar: [0; 2], off: [0; 2], turn: WHITE }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Move {
    /// Point index 0-23, or [`BAR_SENTINEL`].
    pub src: c_int,
    /// Point index 0-23, or [`OFF_SENTINEL`].
    pub dst: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Play {
    pub moves: [Move; MAX_MOVES_PER_PLAY],
    pub num_moves: c_int,
    pub resulting_state: BoardState,
}

impl Default for Play {
    fn default() -> Self {
        Self {
            moves: [Move { src: 0, dst: 0 }; MAX_MOVES_PER_PLAY],
            num_moves: 0,
            resulting_state: BoardState::default(),
        }
    }
}

#[repr(C)]
pub struct NNModel {
    pub num_hidden: c_int,
    pub input_size: c_int,
    pub activation: c_int,
    pub output_mode: c_int,
    pub hidden_sizes: [c_int; NN_MAX_LAYERS],
    pub weight: [*mut c_float; NN_MAX_LAYERS + 1],
    pub bias: [*mut c_float; NN_MAX_LAYERS + 1],
    pub layer_in: [c_int; NN_MAX_LAYERS + 1],
    pub layer_out: [c_int; NN_MAX_LAYERS + 1],
    pub buf_a: *mut c_float,
    pub buf_b: *mut c_float,
}

impl Default for NNModel {
    fn default() -> Self {
        Self {
            num_hidden: 0,
            input_size: 0,
            activation: 0,
            output_mode: 0,
            hidden_sizes: [0; NN_MAX_LAYERS],
            weight: [std::ptr::null_mut(); NN_MAX_LAYERS + 1],
            bias: [std::ptr::null_mut(); NN_MAX_LAYERS + 1],
            layer_in: [0; NN_MAX_LAYERS + 1],
            layer_out: [0; NN_MAX_LAYERS + 1],
            buf_a: std::ptr::null_mut(),
            buf_b: std::ptr::null_mut(),
        }
    }
}

extern "C" {
    pub fn board_init(state: *mut BoardState);
    pub fn board_is_game_over(state: *const BoardState) -> c_int;
    pub fn board_winner(state: *const BoardState) -> c_int;
    pub fn board_switch_turn(state: *mut BoardState);
    pub fn encode_state(state: *const BoardState, features: *mut c_float);

    pub fn get_legal_plays_encoded(
        state: *const BoardState,
        d1: c_int,
        d2: c_int,
        plays: *mut Play,
        max_plays: c_int,
        encoded_features: *mut c_float,
    ) -> c_int;

    pub fn nn_load(model: *mut NNModel, path: *const c_char) -> c_int;
    pub fn nn_free(model: *mut NNModel);
    pub fn nn_forward(model: *const NNModel, input: *const c_float) -> c_float;
    pub fn nn_forward_prob5(
        model: *const NNModel,
        input: *const c_float,
        probs: *mut c_float,
    ) -> c_float;
}
