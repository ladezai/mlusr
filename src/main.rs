use candle_core::{Result};

mod sequential_layers;
mod example_net;
use example_net::{SIN_DATASET_EXAMPLE};

fn main() -> Result<()> {
    SIN_DATASET_EXAMPLE()
}
