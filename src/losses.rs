use ndarray::{ArrayD, Ix2};

pub trait LossFunction {
  fn forward(&self, pred: ArrayD<f64>, target: ArrayD<f64>) -> f64;
  fn backward(&self, pred: ArrayD<f64>, target: ArrayD<f64>) -> ArrayD<f64>;
}

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
  pub fn new() -> Self {
    CrossEntropyLoss {}
  }
}

impl LossFunction for CrossEntropyLoss {
  fn forward(&self, pred: ArrayD<f64>, target: ArrayD<f64>) -> f64 {
    let esp = 1e-15;
    let y_pred = pred.view().into_dimensionality::<Ix2>().unwrap();
    let y_true = target.view().into_dimensionality::<Ix2>().unwrap();
    let log_probs = y_pred.mapv(|x| x.max(esp).min(1.0 - esp).ln());
    let loss = -(&y_true * log_probs);
    loss.sum() / y_pred.shape()[0] as f64
  }

  fn backward(&self, pred: ArrayD<f64>, target: ArrayD<f64>) -> ArrayD<f64> {
    let y_pred = pred.view().into_dimensionality::<Ix2>().unwrap();
    let y_true = target.view().into_dimensionality::<Ix2>().unwrap();
    let grad = -(&y_true / &y_pred.mapv(|x| x.max(1e-15))) / (y_pred.shape()[0] as f64);
    grad.into_dyn()
  }
}
