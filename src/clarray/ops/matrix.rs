use num_traits::{Bounded, Num};
use ocl::{Kernel, OclPrm, core::OclNum};

use crate::clarray::{
  error::Error,
  kernel::{
    clang_type_name,
    matrix::{
      dot_source, elementwise_op_scalar_l_source, elementwise_op_scalar_r_source,
      elementwise_op_source,
    },
    op_to_suffix,
  },
  tensor::{GPUMatrix, OclComputeNum},
};

fn elementwise_op<T>(
  lhs: &GPUMatrix<T>,
  rhs: &GPUMatrix<T>,
  op: &str,
) -> Result<GPUMatrix<T>, Error>
where
  T: OclComputeNum,
{
  if lhs.shape != rhs.shape {
    return Err(crate::clarray::error::Error::MismatchedShape {
      expected: lhs.shape.to_vec(),
      found: rhs.shape.to_vec(),
    });
  }

  let output = GPUMatrix::zeros(lhs.shape, lhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let type_name = clang_type_name(&type_suffix);

  let kernel_name = format!("elementwise_{}_mat_{}", op_to_suffix(&op), type_suffix);
  let program = lhs.env.get_or_compile_program(&kernel_name, || {
    elementwise_op_source(op, &type_name, &type_suffix)
  })?;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(lhs.env.queue.clone())
    .global_work_size(lhs.shape)
    .arg(&lhs.buffer)
    .arg(lhs.strides[0] as i32)
    .arg(lhs.strides[1] as i32)
    .arg(lhs.offset[0] as i32)
    .arg(lhs.offset[1] as i32)
    .arg(&rhs.buffer)
    .arg(rhs.strides[0] as i32)
    .arg(rhs.strides[1] as i32)
    .arg(rhs.offset[0] as i32)
    .arg(rhs.offset[1] as i32)
    .arg(&output.buffer)
    .arg(output.strides[0] as i32)
    .arg(output.strides[1] as i32)
    .arg(output.offset[0] as i32)
    .arg(output.offset[1] as i32)
    .arg(lhs.shape[0] as i32)
    .arg(lhs.shape[1] as i32)
    .build()?;

  unsafe {
    kernel.enq()?;
  }
  Ok(output)
}

fn elementwise_op_l<T>(lhs: &T, rhs: &GPUMatrix<T>, op: &str) -> Result<GPUMatrix<T>, Error>
where
  T: OclComputeNum,
{
  let output = GPUMatrix::zeros(rhs.shape, rhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let type_name = clang_type_name(&type_suffix);

  let kernel_name = format!(
    "elementwise_{}_scalar_l_mat_{}",
    op_to_suffix(&op),
    type_suffix
  );
  let program = rhs.env.get_or_compile_program(&kernel_name, || {
    elementwise_op_scalar_l_source(op, &type_name, &type_suffix)
  })?;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(rhs.env.queue.clone())
    .global_work_size(rhs.shape)
    .arg(lhs)
    .arg(&rhs.buffer)
    .arg(rhs.strides[0] as i32)
    .arg(rhs.strides[1] as i32)
    .arg(rhs.offset[0] as i32)
    .arg(rhs.offset[1] as i32)
    .arg(&output.buffer)
    .arg(output.strides[0] as i32)
    .arg(output.strides[1] as i32)
    .arg(output.offset[0] as i32)
    .arg(output.offset[1] as i32)
    .arg(rhs.shape[0] as i32)
    .arg(rhs.shape[1] as i32)
    .build()?;

  unsafe {
    kernel.enq()?;
  }
  Ok(output)
}

fn elementwise_op_r<T>(lhs: &GPUMatrix<T>, rhs: &T, op: &str) -> Result<GPUMatrix<T>, Error>
where
  T: OclComputeNum,
{
  let output = GPUMatrix::zeros(lhs.shape, lhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let type_name = clang_type_name(&type_suffix);

  let kernel_name = format!(
    "elementwise_{}_scalar_r_mat_{}",
    op_to_suffix(&op),
    type_suffix
  );
  let program = lhs.env.get_or_compile_program(&kernel_name, || {
    elementwise_op_scalar_r_source(op, &type_name, &type_suffix)
  })?;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(lhs.env.queue.clone())
    .global_work_size(lhs.shape)
    .arg(&lhs.buffer)
    .arg(lhs.strides[0] as i32)
    .arg(lhs.strides[1] as i32)
    .arg(lhs.offset[0] as i32)
    .arg(lhs.offset[1] as i32)
    .arg(rhs)
    .arg(&output.buffer)
    .arg(output.strides[0] as i32)
    .arg(output.strides[1] as i32)
    .arg(output.offset[0] as i32)
    .arg(output.offset[1] as i32)
    .arg(lhs.shape[0] as i32)
    .arg(lhs.shape[1] as i32)
    .build()?;

  unsafe {
    kernel.enq()?;
  }
  Ok(output)
}

macro_rules! impl_matrix_op {
  ($trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'a, 'b, T> std::ops::$trait<&'b GPUMatrix<T>> for &'a GPUMatrix<T>
    where
      T: OclComputeNum,
    {
      type Output = Result<GPUMatrix<T>, Error>;
      fn $method(self, rhs: &'b GPUMatrix<T>) -> Self::Output {
        Ok($func(self, rhs, $op_symbol)?)
      }
    }
  };
}

impl_matrix_op!(Add, add, elementwise_op, "+");
impl_matrix_op!(Sub, sub, elementwise_op, "-");
impl_matrix_op!(Mul, mul, elementwise_op, "*");
impl_matrix_op!(Div, div, elementwise_op, "/");
impl_matrix_op!(Rem, rem, elementwise_op, "%");

macro_rules! impl_matrix_op_r {
  ($trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'a, T> std::ops::$trait<T> for &'a GPUMatrix<T>
    where
      T: OclComputeNum,
    {
      type Output = Result<GPUMatrix<T>, Error>;
      fn $method(self, rhs: T) -> Self::Output {
        elementwise_op_r(self, &rhs, $op_symbol)
      }
    }
  };
}
impl_matrix_op_r!(Add, add, elementwise_op_r, "+");
impl_matrix_op_r!(Sub, sub, elementwise_op_r, "-");
impl_matrix_op_r!(Mul, mul, elementwise_op_r, "*");
impl_matrix_op_r!(Div, div, elementwise_op_r, "/");
impl_matrix_op_r!(Rem, rem, elementwise_op_r, "%");

macro_rules! impl_scalar_left {
  ($trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'b> std::ops::$trait<&'b GPUMatrix<f32>> for f32 {
      type Output = Result<GPUMatrix<f32>, Error>;
      fn $method(self, rhs: &'b GPUMatrix<f32>) -> Self::Output {
        $func(&self, rhs, $op_symbol)
      }
    }

    impl<'b> std::ops::$trait<&'b GPUMatrix<f64>> for f64 {
      type Output = Result<GPUMatrix<f64>, Error>;
      fn $method(self, rhs: &'b GPUMatrix<f64>) -> Self::Output {
        $func(&self, rhs, $op_symbol)
      }
    }
  };
}

impl_scalar_left!(Add, add, elementwise_op_l, "+");
impl_scalar_left!(Sub, sub, elementwise_op_l, "-");
impl_scalar_left!(Mul, mul, elementwise_op_l, "*");
impl_scalar_left!(Div, div, elementwise_op_l, "/");
impl_scalar_left!(Rem, rem, elementwise_op_l, "%");

impl<T> GPUMatrix<T>
where
  T: OclComputeNum,
{
  pub fn dot(&self, rhs: &GPUMatrix<T>) -> Result<GPUMatrix<T>, Error> {
    if self.shape[1] != rhs.shape[0] {
      return Err(Error::DotDimensionMismatch {
        a_cols: self.shape[1],
        b_rows: rhs.shape[0],
      });
    }

    let k = self.shape[1];

    let output_shape = [self.shape[0], rhs.shape[1]];
    let output = GPUMatrix::zeros(output_shape, self.env.clone())?;

    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("dot_mat_{}", type_suffix);

    let program = self
      .env
      .get_or_compile_program(&kernel_name, || dot_source(&type_name, &type_suffix))?;

    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size(output_shape)
      .arg(&self.buffer)
      .arg(self.strides[0] as i32)
      .arg(self.strides[1] as i32)
      .arg(self.offset[0] as i32)
      .arg(self.offset[1] as i32)
      .arg(&rhs.buffer)
      .arg(rhs.strides[0] as i32)
      .arg(rhs.strides[1] as i32)
      .arg(rhs.offset[0] as i32)
      .arg(rhs.offset[1] as i32)
      .arg(&output.buffer)
      .arg(output.strides[0] as i32)
      .arg(output.strides[1] as i32)
      .arg(output.offset[0] as i32)
      .arg(output.offset[1] as i32)
      .arg(self.shape[0] as i32)
      .arg(rhs.shape[1] as i32)
      .arg(k as i32)
      .build()?;

    unsafe {
      kernel.enq()?;
    }
    Ok(output)
  }

  pub fn mapv<F>(&self, f: F) -> Result<GPUMatrix<T>, Error>
  where
    F: Fn(T) -> T + Send + Sync,
  {
    let mut data = vec![T::default(); self.buffer.len()];
    self
      .buffer
      .read(&mut data)
      .enq()
      .map_err(|e| Error::OclError(e))?;
    let mapped_data: Vec<T> = data.into_iter().map(f).collect();
    self
      .buffer
      .write(&mapped_data)
      .enq()
      .map_err(|e| Error::OclError(e))?;
    Ok(self.clone())
  }

  pub fn clip(&self, min: T, max: T) -> Result<GPUMatrix<T>, Error> {
    let output = GPUMatrix::zeros(self.shape.clone(), self.env.clone())?;
    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("clip_mat_{}", type_suffix);
    let program = self.env.get_or_compile_program(&kernel_name, || {
      crate::clarray::kernel::matrix::clip_source(&type_name, &type_suffix)
    })?;
    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size(self.shape)
      .arg(&self.buffer)
      .arg(self.strides[0] as i32)
      .arg(self.strides[1] as i32)
      .arg(self.offset[0] as i32)
      .arg(self.offset[1] as i32)
      .arg(&output.buffer)
      .arg(output.strides[0] as i32)
      .arg(output.strides[1] as i32)
      .arg(output.offset[0] as i32)
      .arg(output.offset[1] as i32)
      .arg(min)
      .arg(max)
      .arg(self.shape[0] as i32)
      .arg(self.shape[1] as i32)
      .build()?;
    unsafe {
      kernel.enq()?;
    }
    Ok(output)
  }
}
