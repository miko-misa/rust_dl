use ndarray::{ArrayD, Axis};
use ndarray_stats::QuantileExt;

pub trait Accuracy {
  fn accuracy(&self, pred: &ArrayD<f64>, target: &ArrayD<f64>) -> f64;
}

pub struct OnehotArgmaxAccuracy;
impl OnehotArgmaxAccuracy {
  pub fn new() -> Self {
    OnehotArgmaxAccuracy {}
  }
}

impl Accuracy for OnehotArgmaxAccuracy {
  fn accuracy(&self, pred: &ArrayD<f64>, target: &ArrayD<f64>) -> f64 {
    // 各行のargmax（予測と正解のクラスラベル）を取得
    let pred_labels = pred
      .axis_iter(Axis(0))
      .map(|row| row.argmax().unwrap())
      .collect::<Vec<_>>();

    let true_labels = target
      .axis_iter(Axis(0))
      .map(|row| row.argmax().unwrap())
      .collect::<Vec<_>>();

    // 正解数をカウント
    let correct = pred_labels
      .iter()
      .zip(true_labels.iter())
      .filter(|(p, t)| p == t)
      .count();

    let total = pred.len_of(Axis(0));
    correct as f64 / total as f64
  }
}
