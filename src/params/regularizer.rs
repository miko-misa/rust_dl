use ndarray::ArrayD;

pub trait Regularizer {
  fn apply(&self, grad: ArrayD<f64>, value: ArrayD<f64>) -> ArrayD<f64>;
}

pub struct L2 {
  pub lambda: f64,
}
impl L2 {
  pub fn new(lambda: f64) -> Self {
    L2 { lambda }
  }
}
impl Regularizer for L2 {
  fn apply(&self, grad: ArrayD<f64>, value: ArrayD<f64>) -> ArrayD<f64> {
    grad + self.lambda * 2.0 * value
  }
}

pub struct L1 {
  pub lambda: f64,
}
impl L1 {
  pub fn new(lambda: f64) -> Self {
    L1 { lambda }
  }
}
impl Regularizer for L1 {
  fn apply(&self, grad: ArrayD<f64>, value: ArrayD<f64>) -> ArrayD<f64> {
    let sign = value.mapv(|x| if x > 0.0 { 1.0 } else { -1.0 });
    (grad + self.lambda * sign).into_dyn()
  }
}

pub struct CompositeRegularizer {
  pub regularizers: Vec<Box<dyn Regularizer>>,
}
impl CompositeRegularizer {
  pub fn new(regularizers: Vec<Box<dyn Regularizer>>) -> Self {
    CompositeRegularizer { regularizers }
  }
}
impl Regularizer for CompositeRegularizer {
  fn apply(&self, grad: ArrayD<f64>, value: ArrayD<f64>) -> ArrayD<f64> {
    let mut result = grad;
    for reg in &self.regularizers {
      result = reg.apply(result.clone(), value.clone());
    }
    result
  }
}

pub struct L1L2;
impl L1L2 {
  pub fn new(l1_lambda: f64, l2_lambda: f64) -> CompositeRegularizer {
    CompositeRegularizer {
      regularizers: vec![Box::new(L1::new(l1_lambda)), Box::new(L2::new(l2_lambda))],
    }
  }
}


