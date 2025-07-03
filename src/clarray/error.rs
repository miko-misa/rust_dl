use crate::clarray::error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  #[error("OpenCL error: {0}")]
  OclError(#[from] ocl::Error),

  #[error("Mismatched shape: expected {expected:?}, found {found:?}")]
  MismatchedShape {
    expected: Vec<usize>,
    found: Vec<usize>,
  },

  #[error("Dimension mismatch: a_cols = {a_cols}, b_rows = {b_rows}")]
  DotDimensionMismatch { a_cols: usize, b_rows: usize },

  #[error("Operation requires a contiguous memory layout, but strides were {strides:?}")]
  NotContiguous { strides: [usize; 2] },
}

pub type Result<T> = std::result::Result<T, Error>;
