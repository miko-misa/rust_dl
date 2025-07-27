use rand::rand_core::le;

use crate::{
  clarray::{
    env::env,
    tensor::{DynamicGPUTensor, GPUMatrix, GPUTensor, GPUVector, Tensor},
  },
  params::{
    initializer::{Initializer, ZeroInitializer},
    param::LearnableParameter,
  },
};

pub trait Layer<I, O>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
  O: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<f64, I>) -> GPUTensor<f64, O>;
  fn backward(&mut self, grad: GPUTensor<f64, O>) -> GPUTensor<f64, I>;
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>>;
  fn set_training(&mut self, training: bool);
}

pub struct AffineLayer {
  weight: LearnableParameter<f64>,
  bias: LearnableParameter<f64>,
  training: bool,
  input_cache: Option<GPUTensor<f64, [usize; 2]>>,
}

impl AffineLayer {
  pub fn new<O>(input_dim: usize, output_dim: usize, weight_init: &O) -> Self
  where
    O: Initializer<f64, [usize; 2]>,
  {
    let weight = LearnableParameter::new([input_dim, output_dim], weight_init);
    let bias = LearnableParameter::new([output_dim], &ZeroInitializer);
    AffineLayer {
      weight,
      bias,
      training: true,
      input_cache: None,
    }
  }
}

impl Layer<[usize; 2], [usize; 2]> for AffineLayer {
  fn forward(&mut self, input: GPUMatrix<f64>) -> GPUMatrix<f64> {
    if self.training {
      self.input_cache = Some(input.clone());
    }
    let weight_tensor = self
      .weight
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight to GPUMatrix");
    let bias_tensor = self
      .bias
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 1]>>()
      .expect("Failed to downcast bias to GPUVector");
    let output = &input.dot(weight_tensor).unwrap();
    let output = (output + &bias_tensor.broadcast_matrix(input.shape[0]).unwrap()).unwrap();
    output
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 2]>) -> GPUTensor<f64, [usize; 2]> {
    let x = self
      .input_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.");
    let w = self
      .weight
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight to GPUMatrix");
    self.weight.grads = Box::new(x.t().dot(&grad).unwrap());
    let bias_grad = grad.t().row_sum().unwrap();
    self.bias.grads = Box::new(bias_grad);
    grad.dot(&w.t()).unwrap()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![&mut self.weight, &mut self.bias]
  }

  fn set_training(&mut self, training: bool) {
    self.training = training;
  }
}

pub struct ReLU<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  mask_cache: Option<GPUTensor<f64, I>>,
}

impl<I> ReLU<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  pub fn new() -> Self {
    ReLU { mask_cache: None }
  }
}

impl<I> Layer<I, I> for ReLU<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<f64, I>) -> GPUTensor<f64, I> {
    let mask = input.relu_mask().unwrap();
    self.mask_cache = Some(mask.clone());
    (&input * &mask).unwrap()
  }

  fn backward(&mut self, grad: GPUTensor<f64, I>) -> GPUTensor<f64, I> {
    let mask = self
      .mask_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.");
    (&grad * &mask).unwrap()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![]
  }

  fn set_training(&mut self, _training: bool) {}
}

pub struct Softmax {
  outpu_cache: Option<GPUTensor<f64, [usize; 2]>>,
}

impl Softmax {
  pub fn new() -> Self {
    Softmax { outpu_cache: None }
  }
}

impl Layer<[usize; 2], [usize; 2]> for Softmax {
  fn forward(&mut self, input: GPUMatrix<f64>) -> GPUMatrix<f64> {
    let data = input.to_cpu().unwrap().data;
    // println!("Input data: {:?}", data[0]);
    let max_val = data
      .into_iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap();
    let exp_input = (&input - max_val).unwrap().mapv(|x| x.exp()).unwrap();
    let sum_exp = exp_input
      .row_sum()
      .unwrap()
      .broadcast_matrix(input.shape[1])
      .unwrap()
      .t()
      .clip(1e-12, f64::MAX)
      .unwrap();
    let output = (&exp_input / &sum_exp).unwrap();
    let output = output.contiguous().unwrap();
    self.outpu_cache = Some(output.clone());
    output
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 2]>) -> GPUTensor<f64, [usize; 2]> {
    let y = self
      .outpu_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.");
    let dx = GPUMatrix::zeros(y.shape, env()).unwrap();
    for ((dxi, dyi), yi) in dx.row_iter().zip(grad.row_iter()).zip(y.row_iter()) {
      let diag_y = GPUMatrix::from_diag(&yi, env()).unwrap();
      let yi_mat = yi.broadcast_matrix(1).unwrap();
      let dyi_mat = dyi.broadcast_matrix(1).unwrap();
      let out = dyi_mat
        .dot(&(&diag_y - &yi_mat.t().dot(&yi_mat).unwrap()).unwrap())
        .unwrap();
      let _ = dxi.write(&out.row_iter().nth(0).expect("Iterator is empty"));
    }
    dx
    // grad
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![]
  }

  fn set_training(&mut self, _training: bool) {}
}

pub struct Conv2D {
  weight: LearnableParameter<f64>,
  bias: LearnableParameter<f64>,
  training: bool,
  input_cache: Option<GPUMatrix<f64>>,
  input_shape: [usize; 3],
  output_shape: [usize; 3],
  filter_size: [usize; 3],
  stride: [usize; 3],
  is_pad: bool,
  padding: [i32; 4],
}

impl Conv2D {
  pub fn new<O>(
    input_shape: [usize; 3],
    output_channels: usize,
    filter_size: [usize; 2],
    stride: [usize; 2],
    is_pad: bool,
    weight_init: &O,
  ) -> Self
  where
    O: Initializer<f64, [usize; 2]>,
  {
    let weight = LearnableParameter::new(
      [
        input_shape[0] * filter_size[0] * filter_size[1],
        output_channels,
      ],
      weight_init,
    );
    let bias = LearnableParameter::new([output_channels], &ZeroInitializer);
    let padding = [
      (filter_size[0] - stride[0]) / 2,
      (filter_size[0] - stride[0]) / 2,
      (filter_size[1] - stride[1]) / 2,
      (filter_size[1] - stride[1]) / 2,
    ];
    let output_height = (input_shape[1] + padding[0] + padding[1] - filter_size[0]) / stride[0] + 1;
    let output_width = (input_shape[2] + padding[2] + padding[3] - filter_size[1]) / stride[1] + 1;
    let output_shape = [output_channels, output_height, output_width];
    let filter_size = [input_shape[0], filter_size[0], filter_size[1]];
    let stride = [1, stride[0], stride[1]];
    Conv2D {
      weight,
      bias,
      training: true,
      input_cache: None,
      input_shape,
      output_shape,
      is_pad,
      filter_size,
      stride,
      padding: padding.map(|x| x as i32),
    }
  }

  pub fn set_weight(&mut self, value: &GPUMatrix<f64>) {
    self.weight.value = Box::new(value.clone());
  }
}

impl Layer<[usize; 4], [usize; 4]> for Conv2D {
  fn forward(&mut self, input: GPUTensor<f64, [usize; 4]>) -> GPUTensor<f64, [usize; 4]> {
    // println!("Conv2D start");
    let batch_size = input.shape[0];
    let input = if self.is_pad {
      let p = self.padding;
      input
        .padding([0, 0], [p[0], p[1]], [p[2], p[3]])
        .expect("Failed to apply padding")
    } else {
      input
    };
    // println!("input shape: {:?}", input.shape);
    let x_col: GPUTensor<f64, [usize; 2]> = input
      .im2col(self.filter_size, self.stride)
      .expect("Failed to convert input to column format");
    // println!("x_col shape: {:?}", x_col.shape);
    if self.training {
      self.input_cache = Some(x_col.clone());
    }
    let w = self
      .weight
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight to GPUMatrix");
    // println!("weight shape: {:?}", w.shape);
    let b = self
      .bias
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 1]>>()
      .expect("Failed to downcast bias to GPUVector");
    // println!("bias shape: {:?}", b.shape);
    let output = x_col.dot(w).expect("Failed to perform dot product");
    let b_broadcasted = b
      .broadcast_matrix(batch_size * self.output_shape[1] * self.output_shape[2])
      .expect("Failed to broadcast bias matrix");
    let output = (&output + &b_broadcasted).unwrap();
    let output = output
      .col2im(
        [self.output_shape[0], 1, 1],
        [1, 1, 1],
        [
          batch_size,
          self.output_shape[0],
          self.output_shape[1],
          self.output_shape[2],
        ],
      )
      .expect("Failed to convert column format back to tensor");
    // println!("Conv2D output shape: {:?}", output.shape);
    // println!("Conv2D end");
    output
  }
  fn backward(&mut self, grad: GPUTensor<f64, [usize; 4]>) -> GPUTensor<f64, [usize; 4]> {
    // println!("Conv2D backward start");
    let batch_size = grad.shape[0];
    let x_col = self
      .input_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.");
    let w = self
      .weight
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight to GPUMatrix");
    let dy = grad
      .im2col([self.output_shape[0], 1, 1], [1, 1, 1])
      .expect("Failed to convert grad to column format");
    let b_grad = dy.t().row_sum().unwrap();
    self
      .bias
      .grads
      .as_any_mut()
      .downcast_mut::<GPUTensor<f64, [usize; 1]>>()
      .expect("Failed to downcast bias grads")
      .write(&b_grad)
      .expect("Failed to write bias grads");
    let dx_col = dy.dot(&w.t()).unwrap();
    let dw = x_col.t().dot(&dy).unwrap();
    self
      .weight
      .grads
      .as_any_mut()
      .downcast_mut::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight grads")
      .write(&dw)
      .expect("Failed to write weight grads");
    let dx = dx_col
      .col2im(
        self.filter_size,
        self.stride,
        [
          batch_size,
          self.input_shape[0],
          self.input_shape[1] + (self.padding[0] + self.padding[1]) as usize,
          self.input_shape[2] + (self.padding[2] + self.padding[3]) as usize,
        ],
      )
      .expect("Failed to convert column format back to tensor");
    let dx = if self.is_pad {
      let p = self.padding;
      dx.padding([0, 0], [-p[0], -p[1]], [-p[2], -p[3]])
        .expect("Failed to remove padding")
    } else {
      dx
    };
    // println!("Conv2D backward end");
    dx
  }
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![&mut self.weight, &mut self.bias]
  }
  fn set_training(&mut self, training: bool) {
    self.training = training;
  }
}

pub struct MaxPooling2D {
  input_shape: [usize; 3],
  output_shape: [usize; 3],
  pool_size: [usize; 3],
  stride: [usize; 3],
  mask_cache: Option<GPUMatrix<f64>>,
}
impl MaxPooling2D {
  pub fn new(input_shape: [usize; 3], pool_size: [usize; 2], stride: [usize; 2]) -> Self {
    let output_height = (input_shape[1] - pool_size[0]) / stride[0] + 1;
    let output_width = (input_shape[2] - pool_size[1]) / stride[1] + 1;
    MaxPooling2D {
      input_shape,
      output_shape: [input_shape[0], output_height, output_width],
      pool_size: [1, pool_size[0], pool_size[1]],
      stride: [1, stride[0], stride[1]],
      mask_cache: None,
    }
  }
}

impl Layer<[usize; 4], [usize; 4]> for MaxPooling2D {
  fn forward(&mut self, input: GPUTensor<f64, [usize; 4]>) -> GPUTensor<f64, [usize; 4]> {
    // println!("MaxPooling2D start");
    let batch_size = input.shape[0];
    let x_col = input
      .im2col(self.pool_size, self.stride)
      .expect("Failed to convert input to column format");
    let mask = x_col.row_max_mask().unwrap();
    if self.mask_cache.is_none() {
      self.mask_cache = Some(mask.clone());
    }
    let output_vec = (&x_col * &mask).unwrap().row_sum().unwrap();
    let mut output = GPUTensor::<f64, [usize; 4]>::zeros(
      [
        batch_size,
        self.output_shape[0],
        self.output_shape[1],
        self.output_shape[2],
      ],
      env(),
    )
    .unwrap();
    output.write_buffer(&output_vec.buffer).unwrap();
    // println!("MaxPooling2D end");
    output
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 4]>) -> GPUTensor<f64, [usize; 4]> {
    // println!("MaxPooling2D backward start");
    let dy_col_vec = grad
      .im2col([1, 1, 1], [1, 1, 1])
      .expect("Failed to convert grad to column format")
      .row_sum()
      .unwrap();
    let dy_col_copied = dy_col_vec
      .broadcast_matrix(self.pool_size[1] * self.pool_size[2])
      .unwrap()
      .t();
    let mask = self
      .mask_cache
      .as_ref()
      .expect("Mask cache is not set. Ensure forward is called before backward.");
    let dx_col = (&dy_col_copied * mask).unwrap();
    let dx = dx_col
      .col2im(
        self.pool_size,
        self.stride,
        [
          grad.shape[0],
          self.input_shape[0],
          self.input_shape[1],
          self.input_shape[2],
        ],
      )
      .unwrap();
    // println!("MaxPooling2D backward end");
    dx
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![]
  }

  fn set_training(&mut self, _training: bool) {}
}

pub struct Flatten<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  input_shape: Option<I>,
}

impl<I> Flatten<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  pub fn new() -> Self {
    Flatten { input_shape: None }
  }
}
impl<I> Layer<I, [usize; 2]> for Flatten<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<f64, I>) -> GPUTensor<f64, [usize; 2]> {
    self.input_shape = Some(input.shape.clone());
    // println!("input shape: {:?}", input.shape);
    let batch_size = input.shape.as_ref()[0];
    let len = input.len() / batch_size;
    let mut output = GPUMatrix::zeros([batch_size, len], env()).unwrap();
    output.write_buffer(&input.buffer).unwrap();
    // println!("Flatten output shape: {:?}", output.shape);
    output
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 2]>) -> GPUTensor<f64, I> {
    let batch_size = grad.shape[0];
    let mut output_shape = self
      .input_shape
      .as_ref()
      .expect("Input shape is not set. Ensure forward is called before backward.")
      .clone();
    output_shape.as_mut()[0] = batch_size;
    let mut output = GPUTensor::<f64, I>::zeros(output_shape, env()).unwrap();
    output.write_buffer(&grad.buffer).unwrap();
    output
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![]
  }

  fn set_training(&mut self, _training: bool) {}
}
