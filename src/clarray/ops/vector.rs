use num_traits::Num;
use ocl::{Kernel, OclPrm, core::OclNum};

use crate::clarray::{
  error::Error,
  kernel::{
    clang_type_name, op_to_suffix,
    vector::{
      broadcast_matrix_source, elementwise_op_scalar_l_source, elementwise_op_scalar_r_source,
      elementwise_op_source, sum_source,
    },
  },
  tensor::{GPUMatrix, GPUVector, OclComputeNum},
};

fn elementwise_op<T>(
  lhs: &GPUVector<T>,
  rhs: &GPUVector<T>,
  op: &str,
) -> Result<GPUVector<T>, Error>
where
  T: OclComputeNum,
{
  if lhs.shape != rhs.shape {
    return Err(crate::clarray::error::Error::MismatchedShape {
      expected: lhs.shape.to_vec(),
      found: rhs.shape.to_vec(),
    });
  }

  let output = GPUVector::zeros(lhs.shape, lhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let kernel_name = format!("elementwise_{}_vec_{}", op_to_suffix(&op), type_suffix);
  let program = lhs.env.get_or_compile_program(&type_suffix)?;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(lhs.env.queue.clone())
    .global_work_size(lhs.shape)
    .arg(&lhs.buffer)
    .arg(lhs.strides[0] as u64)
    .arg(lhs.offset[0] as u64)
    .arg(&rhs.buffer)
    .arg(rhs.strides[0] as u64)
    .arg(rhs.offset[0] as u64)
    .arg(&output.buffer)
    .arg(output.strides[0] as u64)
    .arg(output.offset[0] as u64)
    .arg(lhs.shape[0] as u64)
    .build()?;

  unsafe {
    kernel.enq()?;
  }
  Ok(output)
}

fn elementwise_op_l<T>(lhs: &T, rhs: &GPUVector<T>, op: &str) -> Result<GPUVector<T>, Error>
where
  T: OclComputeNum,
{
  let output = GPUVector::zeros(rhs.shape, rhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();

  let kernel_name = format!(
    "elementwise_{}_scalar_l_vec_{}",
    op_to_suffix(&op),
    type_suffix
  );

  let program = rhs.env.get_or_compile_program(&type_suffix)?;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(rhs.env.queue.clone())
    .global_work_size(rhs.shape)
    .arg(lhs)
    .arg(&rhs.buffer)
    .arg(rhs.strides[0] as u64)
    .arg(rhs.offset[0] as u64)
    .arg(&output.buffer)
    .arg(output.strides[0] as u64)
    .arg(output.offset[0] as u64)
    .arg(rhs.shape[0] as u64)
    .build()?;

  unsafe {
    kernel.enq()?;
  }
  Ok(output)
}

fn elementwise_op_r<T>(lhs: &GPUVector<T>, rhs: &T, op: &str) -> Result<GPUVector<T>, Error>
where
  T: OclComputeNum,
{
  let output = GPUVector::zeros(lhs.shape, lhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let kernel_name = format!(
    "elementwise_{}_scalar_r_vec_{}",
    op_to_suffix(&op),
    type_suffix
  );
  let program = lhs.env.get_or_compile_program(&type_suffix)?;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(lhs.env.queue.clone())
    .global_work_size(lhs.shape)
    .arg(&lhs.buffer)
    .arg(lhs.strides[0] as u64)
    .arg(lhs.offset[0] as u64)
    .arg(rhs)
    .arg(&output.buffer)
    .arg(output.strides[0] as u64)
    .arg(output.offset[0] as u64)
    .arg(lhs.shape[0] as u64)
    .build()?;

  unsafe {
    kernel.enq()?;
  }
  Ok(output)
}

macro_rules! impl_matrix_op_vec {
  ($trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'a, 'b, T> std::ops::$trait<&'b GPUVector<T>> for &'a GPUVector<T>
    where
      T: OclComputeNum,
    {
      type Output = Result<GPUVector<T>, Error>;
      fn $method(self, rhs: &'b GPUVector<T>) -> Self::Output {
        Ok($func(self, rhs, $op_symbol)?)
      }
    }
  };
}

/*
impl_matrix_op_vec!(Add, add, elementwise_op, "+");
impl_matrix_op_vec!(Sub, sub, elementwise_op, "-");
impl_matrix_op_vec!(Mul, mul, elementwise_op, "*");
impl_matrix_op_vec!(Div, div, elementwise_op, "/");
impl_matrix_op_vec!(Rem, rem, elementwise_op, "%");
*/

macro_rules! impl_matrix_op_r_vec {
  ($trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'a, T> std::ops::$trait<T> for &'a GPUVector<T>
    where
      T: OclComputeNum,
    {
      type Output = Result<GPUVector<T>, Error>;
      fn $method(self, rhs: T) -> Self::Output {
        elementwise_op_r(self, &rhs, $op_symbol)
      }
    }
  };
}
impl_matrix_op_r_vec!(Add, add, elementwise_op_r, "+");
impl_matrix_op_r_vec!(Sub, sub, elementwise_op_r, "-");
impl_matrix_op_r_vec!(Mul, mul, elementwise_op_r, "*");
impl_matrix_op_r_vec!(Div, div, elementwise_op_r, "/");
impl_matrix_op_r_vec!(Rem, rem, elementwise_op_r, "%");

macro_rules! impl_scalar_l_vec {
  ($trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'b> std::ops::$trait<&'b GPUVector<f32>> for f32 {
      type Output = Result<GPUVector<f32>, Error>;
      fn $method(self, rhs: &'b GPUVector<f32>) -> Self::Output {
        $func(&self, rhs, $op_symbol)
      }
    }

    impl<'b> std::ops::$trait<&'b GPUVector<f64>> for f64 {
      type Output = Result<GPUVector<f64>, Error>;
      fn $method(self, rhs: &'b GPUVector<f64>) -> Self::Output {
        $func(&self, rhs, $op_symbol)
      }
    }
  };
}

impl_scalar_l_vec!(Add, add, elementwise_op_l, "+");
impl_scalar_l_vec!(Sub, sub, elementwise_op_l, "-");
impl_scalar_l_vec!(Mul, mul, elementwise_op_l, "*");
impl_scalar_l_vec!(Div, div, elementwise_op_l, "/");
impl_scalar_l_vec!(Rem, rem, elementwise_op_l, "%");

impl<T> GPUVector<T>
where
  T: OclComputeNum,
{
  pub fn broadcast_matrix(&self, rows: usize) -> Result<GPUMatrix<T>, Error> {
    let output = GPUMatrix {
      buffer: self.buffer.clone(),
      strides: [0, self.strides[0]],
      offset: [0, self.offset[0]],
      shape: [rows, self.shape[0]],
      env: self.env.clone(),
    };
    Ok(output)
  }

  /*
  pub fn broadcast_matrix(&self, rows: usize) -> Result<GPUMatrix<T>, Error> {
    let output = GPUMatrix::zeros([rows, self.shape[0]], self.env.clone())?;

    let type_suffix = std::any::type_name::<T>();

    let kernel_name = format!("broadcast_matrix_vec_{}", type_suffix);
    let program = self.env.get_or_compile_program(&type_suffix)?;
    let kernel = ocl::Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size([rows, self.shape[0]])
      .arg(&self.buffer)
      .arg(self.strides[0] as u64)
      .arg(self.offset[0] as u64)
      .arg(&output.buffer)
      .arg(output.strides[0] as u64)
      .arg(output.strides[1] as u64)
      .arg(output.offset[0] as u64)
      .arg(output.offset[1] as u64)
      .arg(rows as u64)
      .arg(self.shape[0] as u64)
      .build()?;

    unsafe {
      kernel.enq()?;
    }
    Ok(output)
  }*/

  pub fn sum(&self) -> Result<T, Error> {
    let type_suffix = std::any::type_name::<T>();

    let kernel_name = format!("sum_vec_{}", type_suffix);
    let program = self.env.get_or_compile_program(&type_suffix)?;

    let output_buffer = ocl::Buffer::<T>::builder()
      .queue(self.env.queue.clone())
      .len(1)
      .build()?;

    let kernel = ocl::Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size(1)
      .arg(&self.buffer)
      .arg(self.strides[0] as u64)
      .arg(self.offset[0] as u64)
      .arg(&output_buffer)
      .arg(self.shape[0] as u64)
      .build()?;

    unsafe {
      kernel.enq()?;
    }

    let mut output_data = vec![T::default()];
    output_buffer.read(&mut output_data).enq()?;
    Ok(output_data[0])
  }
}
