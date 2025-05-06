use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::fs::File;

pub fn load_csv_to_ndarray(
  path: &str,
  has_headers: bool,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
  let file = File::open(path)?;
  let mut reader = ReaderBuilder::new()
    .has_headers(has_headers)
    .from_reader(file);
  let array = reader.deserialize_array2_dynamic()?;
  Ok(array)
}
