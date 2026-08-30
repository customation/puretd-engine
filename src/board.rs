//! Turning a GnuBG position into a PureTD board.
//!
//! This is the join between two independent codebases and the one place where a
//! mistake is invisible: a mirrored board is a perfectly legal position, so the
//! engine would answer confidently about a game nobody is playing. The mapping
//! is therefore asserted against PureTD's own `board_init` in the tests below
//! rather than reasoned about in a comment.
//!
//! The two conventions turn out to line up exactly:
//!
//! * `bep_protocol::gnubg_ids::Board` is a 26-int array in the MOVER's view —
//!   `board[1..=24]` are the points counted from the mover's home, positive for
//!   the mover and negative for the opponent, `board[25]` is the mover's bar and
//!   `board[0]` the opponent's.
//! * PureTD's `BoardState.points[0..24]` is the same numbering shifted by one,
//!   positive for WHITE. `board_init` puts WHITE's two back checkers on
//!   `points[23]` and comments it "point 24".
//!
//! So the mover becomes WHITE and `points[i] = board[i + 1]`. Nothing is
//! mirrored, reversed or renumbered.
//!
//! Neither side's borne-off count is carried by a Position ID — it is implied by
//! the fifteen checkers that are not on the board or the bar, which is how gnubg
//! stores it and why a position with checkers missing is not an error.

use bep_protocol::gnubg_ids::Board;

use crate::ffi::{BoardState, BLACK, NUM_POINTS, WHITE};

/// Every player has fifteen checkers; what is not on the board or the bar is off.
const CHECKERS_PER_PLAYER: i32 = 15;

/// The mover's board, as PureTD wants it, with the mover on the WHITE side.
pub fn to_puretd(board: &Board) -> BoardState {
    let mut state = BoardState::default();

    let mut mover_on_board = 0;
    let mut opponent_on_board = 0;
    for i in 0..NUM_POINTS {
        let checkers = board[i + 1];
        state.points[i] = checkers;
        if checkers > 0 {
            mover_on_board += checkers;
        } else {
            opponent_on_board += -checkers;
        }
    }

    state.bar[WHITE as usize] = board[25];
    state.bar[BLACK as usize] = board[0];

    // Clamped at zero rather than trusted: a malformed id can describe more
    // than fifteen checkers, and a negative "off" would propagate into the
    // feature encoding as a plausible number instead of a refusal.
    state.off[WHITE as usize] =
        (CHECKERS_PER_PLAYER - mover_on_board - board[25]).max(0);
    state.off[BLACK as usize] =
        (CHECKERS_PER_PLAYER - opponent_on_board - board[0]).max(0);

    state.turn = WHITE;
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use bep_protocol::gnubg_ids::decode_position_id;

    /// The opening position's Position ID, as gnubg writes it.
    const OPENING: &str = "4HPwATDgc/ABMA";

    fn board_init() -> BoardState {
        let mut state = BoardState::default();
        // SAFETY: writes a fully-owned, correctly-sized BoardState.
        unsafe { crate::ffi::board_init(&mut state) };
        state
    }

    /// The mapping is right if converting gnubg's opening position produces
    /// exactly what PureTD builds for itself. Anything mirrored, renumbered or
    /// shifted by one shows up here as a mismatched point.
    #[test]
    fn the_opening_position_matches_puretds_own_board() {
        let decoded = decode_position_id(OPENING).expect("opening id decodes");
        let converted = to_puretd(&decoded);
        let native = board_init();

        assert_eq!(
            converted.points, native.points,
            "converted points differ from board_init"
        );
        assert_eq!(converted.bar, native.bar, "bar differs");
        assert_eq!(converted.off, native.off, "off differs");
        assert_eq!(converted.turn, native.turn, "turn differs");
    }

    /// Fifteen a side, always — the count the ids never carry.
    #[test]
    fn borne_off_checkers_are_derived_not_read() {
        let decoded = decode_position_id(OPENING).expect("opening id decodes");
        let state = to_puretd(&decoded);
        let white: i32 = state.points.iter().filter(|p| **p > 0).sum::<i32>()
            + state.bar[WHITE as usize]
            + state.off[WHITE as usize];
        let black: i32 = state.points.iter().filter(|p| **p < 0).map(|p| -p).sum::<i32>()
            + state.bar[BLACK as usize]
            + state.off[BLACK as usize];
        assert_eq!(white, CHECKERS_PER_PLAYER);
        assert_eq!(black, CHECKERS_PER_PLAYER);
    }
}
