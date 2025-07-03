use num_traits::Num;
use ocl::{Kernel, OclPrm, core::OclNum};

use crate::clarray::{
  error::Error,
  kernel::{
    clang_type_name, dot_source, elementwise_op_scalar_l_source, elementwise_op_scalar_r_source,
    elementwise_op_source, op_to_suffix,
  },
  tensor::GPUMatrix,
};

fn elementwise_op<T>(
  lhs: &GPUMatrix<T>,
  rhs: &GPUMatrix<T>,
  op: &str,
) -> Result<GPUMatrix<T>, Error>
where
  T: OclPrm + Num + Copy + Default + OclNum,
{
  if lhs.shape != rhs.shape {
    return Err(crate::clarray::error::Error::MismatchedShape {
      expected: lhs.shape.to_vec(),
      found: rhs.shape.to_vec(),
    });
  }

  let output = GPUMatrix::new(lhs.shape, lhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let type_name = clang_type_name(&type_suffix);

  let kernel_name = format!("elementwise_{}_{}", op_to_suffix(&op), type_suffix);
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
  T: OclPrm + Num + Copy + Default + OclNum,
{
  let output = GPUMatrix::new(rhs.shape, rhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let type_name = clang_type_name(&type_suffix);

  let kernel_name = format!("elementwise_{}_scalar_l_{}", op_to_suffix(&op), type_suffix);
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
  T: OclPrm + Num + Copy + Default + OclNum,
{
  let output = GPUMatrix::new(lhs.shape, lhs.env.clone())?;

  let type_suffix = std::any::type_name::<T>();
  let type_name = clang_type_name(&type_suffix);

  let kernel_name = format!("elementwise_{}_scalar_r_{}", op_to_suffix(&op), type_suffix);
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
      T: OclPrm + Num + Copy + Default + OclNum,
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
      T: OclPrm + Num + Copy + Default + OclNum,
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
  ($ty:ty, $trait:ident, $method:ident, $func:ident, $op_symbol:expr) => {
    impl<'b> std::ops::$trait<&'b GPUMatrix<$ty>> for $ty {
      type Output = Result<GPUMatrix<$ty>, Error>;
      fn $method(self, rhs: &'b GPUMatrix<$ty>) -> Self::Output {
        $func(&self, rhs, $op_symbol)
      }
    }
  };
}

impl_scalar_left!(f32, Add, add, elementwise_op_l, "+");
impl_scalar_left!(f64, Add, add, elementwise_op_l, "+");
impl_scalar_left!(f32, Sub, sub, elementwise_op_l, "-");
impl_scalar_left!(f64, Sub, sub, elementwise_op_l, "-");
impl_scalar_left!(f32, Mul, mul, elementwise_op_l, "*");
impl_scalar_left!(f64, Mul, mul, elementwise_op_l, "*");
impl_scalar_left!(f32, Div, div, elementwise_op_l, "/");
impl_scalar_left!(f64, Div, div, elementwise_op_l, "/");
impl_scalar_left!(f32, Rem, rem, elementwise_op_l, "%");
impl_scalar_left!(f64, Rem, rem, elementwise_op_l, "%");

impl<T> GPUMatrix<T>
where
  T: OclPrm + Num + Copy + Default + OclNum,
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
    let output = GPUMatrix::new(output_shape, self.env.clone())?;

    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("dot_{}", type_suffix);

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
}
