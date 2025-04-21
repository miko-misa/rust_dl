use ndarray::{
  Array1, Array2, ArrayBase, ArrayD, Axis, Dim, Ix1, Ix2, IxDyn, IxDynImpl, OwnedRepr,
};

pub struct LearnableParam<D = IxDynImpl> {
  pub value: ArrayBase<OwnedRepr<f64>, D>,
  pub grads: ArrayBase<OwnedRepr<f64>, D>,
}

pub trait Layer<D = IxDynImpl> {
  fn forward(&mut self, input: ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D>;
  fn backward(&mut self, grad: ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D>;
}

// Affine2D layer
pub struct Affine2D {
  pub input_dim: usize,
  pub output_dim: usize,
  weights: LearnableParam<Ix2>,
  bias: LearnableParam<Ix1>,
  input: ArrayBase<OwnedRepr<f64>, Ix2>,
}

impl Affine2D {
  pub fn new(input_dim: usize, output_dim: usize) -> Self {
    let weights = LearnableParam::<Ix2> {
      value: Array2::zeros((input_dim, output_dim)),
      grads: Array2::zeros((input_dim, output_dim)),
    };
    let bias = LearnableParam::<Ix1> {
      value: Array1::zeros(output_dim),
      grads: Array1::zeros(output_dim),
    };
    Self {
      input_dim,
      output_dim,
      weights,
      bias,
      input: Array2::zeros((1, input_dim)),
    }
  }
}

impl Layer<Ix2> for Affine2D {
  fn forward(&mut self, input: ArrayBase<OwnedRepr<f64>, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    self.input = input.clone();
    let output = input.dot(&self.weights.value) + &self.bias.value;
    output
  }

  fn backward(&mut self, grad: ArrayBase<OwnedRepr<f64>, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let grad_input = grad.dot(&self.weights.value.t());
    self.weights.grads = self.input.t().dot(&grad);
    self.bias.grads = grad.sum_axis(Axis(0));
    grad_input
  }
}

// Softmax layer
pub struct Softmax {
  pub input_dim: usize,
  pub output_dim: usize,
  pub output: ArrayBase<OwnedRepr<f64>, Ix2>,
}
impl Softmax {
  pub fn new(input_dim: usize, output_dim: usize) -> Self {
    Self {
      input_dim,
      output_dim,
      output: Array2::zeros((1, input_dim)),
    }
  }
}

impl Layer<Ix2> for Softmax {
  fn forward(&mut self, input: ArrayBase<OwnedRepr<f64>, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let max = input.map_axis(Axis(1), |x| {
      x.fold(f64::NEG_INFINITY, |arg0: f64, other: &f64| {
        f64::max(arg0, *other)
      })
    });
    let exp_input = (input - &max).mapv(f64::exp);
    let sum_exp = exp_input.sum_axis(Axis(1)).insert_axis(Axis(1));
    let output = exp_input / sum_exp;
    self.output = output.clone();
    output
  }

  fn backward(&mut self, grad: ArrayBase<OwnedRepr<f64>, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let mut grad_input = Array2::<f64>::zeros(self.output.raw_dim());

    for ((mut grad_input_row, softmax_row), grad_output_row) in grad_input
      .outer_iter_mut()
      .zip(self.output.outer_iter())
      .zip(grad.outer_iter())
    {
      // Softmaxの出力ベクトルを列ベクトルに変換
      let y = softmax_row.to_owned();
      let y_col = y.clone().insert_axis(Axis(1));
      let y_row = y.clone().insert_axis(Axis(0));

      // ヤコビアン行列の計算: diag(y) - y * y^T
      let jacobian = Array2::from_diag(&y) - &y_col.dot(&y_row);

      // 入力に対する勾配の計算: J^T * grad_output
      let grad = jacobian.dot(&grad_output_row.to_owned());

      // 計算結果をgrad_inputに格納
      grad_input_row.assign(&grad);
    }

    grad_input
  }
}

// ReLU layer
pub struct ReLU {
  pub input_dim: usize,
  pub output_dim: usize,
  pub input: ArrayBase<OwnedRepr<f64>, Ix2>,
}

impl ReLU {
  pub fn new(input_dim: usize, output_dim: usize) -> Self {
    Self {
      input_dim,
      output_dim,
      input: Array2::zeros((1, input_dim)),
    }
  }
}

impl Layer<Ix2> for ReLU {
  fn forward(&mut self, input: ArrayBase<OwnedRepr<f64>, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    self.input = input.clone();
    let output = input.mapv(|x| f64::max(0.0, x));
    output
  }

  fn backward(&mut self, grad: ArrayBase<OwnedRepr<f64>, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let grad_input = grad * self.input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
    grad_input
  }
}
