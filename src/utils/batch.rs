use ndarray::{Array2, Axis, s};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};

pub fn create_batches(
  data: &Array2<f64>,
  label_col: usize,
  batch_size: usize,
) -> Vec<(Array2<f64>, Array2<f64>)> {
  let n_rows = data.nrows();
  let n_cols = data.ncols();

  // インデックスのシャッフル
  let mut indices: Vec<usize> = (0..n_rows).collect();
  indices.shuffle(&mut thread_rng());

  // シャッフルされたデータの作成
  let shuffled_data = data.select(Axis(0), &indices);

  // 特徴量とラベルの分割
  let mut features = Vec::new();
  let mut labels = Vec::new();
  for row in shuffled_data.rows() {
    let mut feature_row = Vec::new();
    let mut label_value = 0.0;
    for (i, &val) in row.iter().enumerate() {
      if i == label_col {
        label_value = val;
      } else {
        feature_row.push(val);
      }
    }
    features.push(feature_row);
    labels.push(vec![label_value]);
  }

  // 特徴量とラベルの配列への変換
  let feature_array = Array2::from_shape_vec(
    (n_rows, n_cols - 1),
    features.into_iter().flatten().collect(),
  )
  .unwrap();
  let label_array =
    Array2::from_shape_vec((n_rows, 1), labels.into_iter().flatten().collect()).unwrap();

  // バッチの作成
  let mut batches = Vec::new();
  for i in (0..n_rows).step_by(batch_size) {
    let end = usize::min(i + batch_size, n_rows);
    let feature_batch = feature_array.slice(s![i..end, ..]).to_owned();
    let label_batch = label_array.slice(s![i..end, ..]).to_owned();
    batches.push((feature_batch, label_batch));
  }

  batches
}

pub fn one_hot_encode(labels: &Array2<f64>, num_classes: usize) -> Array2<f64> {
  let n_rows = labels.nrows();
  let mut one_hot = Array2::<f64>::zeros((n_rows, num_classes));
  for (i, row) in labels.outer_iter().enumerate() {
    let class_idx = row[0] as usize;
    if class_idx < num_classes {
      one_hot[[i, class_idx]] = 1.0;
    }
  }
  one_hot
}
