mod obs2;

use hsbg_engine::engine::Game;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn observe<'py>(
    py: Python<'py>,
    state: &str,
    seat: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let observation = py.detach(|| -> PyResult<_> {
        let game: Game =
            serde_json::from_str(state).map_err(|error| PyValueError::new_err(error.to_string()))?;
        game.check_loaded().map_err(PyValueError::new_err)?;
        let player = game
            .players
            .get(seat)
            .ok_or_else(|| PyValueError::new_err("seat out of range"))?;
        if player.sees_all_boards {
            return Err(PyValueError::new_err("privileged opponent view is forbidden"));
        }
        Ok(obs2::observe2_view(&game, seat, false))
    })?;
    let floats: Vec<u8> = observation.floats.iter().flat_map(|x| x.to_le_bytes()).collect();
    let cards: Vec<u8> = observation.cards.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok((PyBytes::new(py, &floats), PyBytes::new(py, &cards)))
}

#[pymodule]
fn _asd_obs2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(observe, m)?)?;
    m.add("OBS2_FLOATS", obs2::OBS2_FLOATS)?;
    m.add("OBS2_CARDS", obs2::OBS2_CARDS)?;
    m.add("FLOAT_BLOCKS", obs2::FLOAT_BLOCKS.to_vec())?;
    m.add("CARD_BLOCKS", obs2::CARD_BLOCKS.to_vec())?;
    m.add("X_MINION", obs2::X_MINION)?;
    m.add("OPP_SUM_F", obs2::OPP_SUM_F)?;
    m.add("K_ATT_BOARD", obs2::K_ATT_BOARD)?;
    m.add("K_ATT_ITEM", obs2::K_ATT_ITEM)?;
    m.add("K_ATT_OPP", obs2::K_ATT_OPP)?;
    Ok(())
}
