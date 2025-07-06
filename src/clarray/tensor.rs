use std::{any::Any, f32::consts, fmt::Debug, sync::Arc};

use num_traits::{Bounded, Float, FromPrimitive, Num, ToPrimitive};
use ocl::{Buffer, Device, Kernel, OclPrm, core::OclNum, flags};

use crate::clarray::{
  env::{GPUEnv, env},
  error::Error,
  kernel::{
    clang_type_name,
    matrix::{repeak_source, row_sum_source},
  },
};

pub trait OclComputeNum:
  OclPrm + Num + Copy + Debug + Default + OclNum + FromPrimitive + ToPrimitive + Bounded + Float
{
}

impl OclComputeNum for f32 {}
impl OclComputeNum for f64 {}

pub struct Tensor<T, D>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  pub shape: D,
  pub data: Vec<T>,
  pub offset: D,
}

pub type Vector<T> = Tensor<T, [usize; 1]>;
pub type Matrix<T> = Tensor<T, [usize; 2]>;

impl<T> Matrix<T>
where
  T: OclComputeNum,
{
  pub fn from_vec(shape: [usize; 2], data: Vec<T>) -> Result<Self, Error> {
    if shape[0] * shape[1] != data.len() {
      return Err(Error::MismatchedShape {
        expected: vec![shape[0], shape[1]],
        found: vec![data.len()],
      });
    }
    Ok(Matrix {
      shape,
      data,
      offset: [0, 0],
    })
  }

  pub fn to_gpu(&self) -> Result<GPUTensor<T, [usize; 2]>, Error> {
    let env = env();
    let buffer = Buffer::<T>::builder()
      .queue(env.queue.clone())
      .len(self.data.len())
      .copy_host_slice(&self.data)
      .build()?;

    Ok(GPUTensor {
      buffer: buffer,
      shape: self.shape.clone(),
      strides: [self.shape[1], 1],
      offset: self.offset.clone(),
      env,
    })
  }
}

#[derive(Clone)]
pub struct GPUTensor<T, D>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  pub(crate) buffer: Buffer<T>,
  pub shape: D,
  pub strides: D,
  pub offset: D,
  pub(crate) env: Arc<GPUEnv>,
}

pub trait DynamicGPUTensor<T>
where
  T: OclComputeNum,
{
  fn as_any(&self) -> &dyn Any;
  fn as_any_mut(&mut self) -> &mut dyn Any;
  fn rank(&self) -> usize;
  fn print_info(&self);
}
impl<T, D> DynamicGPUTensor<T> for GPUTensor<T, D>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone + 'static,
{
  fn as_any(&self) -> &dyn Any {
    self
  }
  fn as_any_mut(&mut self) -> &mut dyn Any {
    self
  }
  fn rank(&self) -> usize {
    self.shape.as_ref().len()
  }

  fn print_info(&self) {
    println!(
      "GPUTensor<Rank: {}, Shape: {:?}, Type: {}>",
      self.rank(),
      self.shape.as_ref(),
      std::any::type_name::<T>()
    );
  }
}

pub type GPUVector<T> = GPUTensor<T, [usize; 1]>;
pub type GPUMatrix<T> = GPUTensor<T, [usize; 2]>;

impl<T, D> GPUTensor<T, D>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  pub fn zeros(shape: D, env: Arc<GPUEnv>) -> Result<Self, Error> {
    Self::from_vec(
      shape.clone(),
      vec![T::zero(); shape.as_ref().iter().product()],
      env,
    )
  }

  pub fn from_vec(shape: D, data: Vec<T>, env: Arc<GPUEnv>) -> Result<Self, Error> {
    if shape.as_ref().iter().product::<usize>() != data.len() {
      return Err(Error::MismatchedShape {
        expected: shape.as_ref().to_vec(),
        found: vec![data.len()],
      });
    }

    let buffer = Buffer::<T>::builder()
      .queue(env.queue.clone())
      .len(data.len())
      .copy_host_slice(&data)
      .build()?;

    let mut stride = shape.clone();
    for i in 0..stride.as_ref().len() {
      if i == 0 {
        stride.as_mut()[i] = 1;
      } else {
        stride.as_mut()[i] = stride.as_ref()[i - 1] * shape.as_ref()[i];
      }
    }

    Ok(GPUTensor {
      buffer,
      shape: shape.clone(),
      strides: stride,
      offset: {
        let mut offset = shape.clone();
        offset.as_mut().iter_mut().for_each(|x| *x = 0);
        offset
      },
      env,
    })
  }

  pub fn len(&self) -> usize {
    self.shape.as_ref().iter().product()
  }
}

impl<T> GPUVector<T>
where
  T: OclComputeNum,
{
  pub fn to_cpu(&self) -> Result<Vector<T>, Error> {
    let mut data = vec![T::default(); self.len()];
    self
      .buffer
      .read(&mut data)
      .enq()
      .map_err(|e| Error::OclError(e))?;
    Ok(Vector {
      shape: [self.len()],
      data,
      offset: [0],
    })
  }

  pub fn write(&self, data: &GPUVector<T>) -> Result<(), Error> {
    if data.shape != self.shape {
      return Err(Error::MismatchedShape {
        expected: vec![self.shape[0]],
        found: vec![data.shape[0]],
      });
    }

    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("write_vec_{}", type_suffix);

    let program = self.env.get_or_compile_program(&kernel_name, || {
      crate::clarray::kernel::vector::write_source(&type_name, &type_suffix)
    })?;

    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size(self.shape)
      .arg(&data.buffer)
      .arg(data.strides[0] as i32)
      .arg(data.offset[0] as i32)
      .arg(&self.buffer)
      .arg(self.strides[0] as i32)
      .arg(self.offset[0] as i32)
      .arg(self.len() as i32)
      .build()?;

    unsafe {
      kernel.enq()?;
    }

    Ok(())
  }
}

impl<T> GPUMatrix<T>
where
  T: OclComputeNum,
{
  pub fn rows(&self) -> usize {
    self.shape[0]
  }

  pub fn cols(&self) -> usize {
    self.shape[1]
  }

  pub fn t(&self) -> GPUMatrix<T> {
    GPUMatrix {
      buffer: self.buffer.clone(),
      shape: [self.shape[1], self.shape[0]],
      strides: [self.strides[1], self.strides[0]],
      offset: [self.offset[1], self.offset[0]],
      env: self.env.clone(),
    }
  }

  pub fn is_contiguous(&self) -> bool {
    self.strides == [self.cols(), 1]
  }

  pub fn contiguous(&self) -> Result<Self, Error> {
    if self.is_contiguous() {
      Ok(self.clone())
    } else {
      let output_buffer = Buffer::<T>::builder()
        .queue(self.env.queue.clone())
        .len(self.len())
        .build()?;

      let type_suffix = std::any::type_name::<T>();
      let type_name = clang_type_name(type_suffix);
      let kernel_key = format!("repeak_mat_{}", type_suffix);
      let program = self
        .env
        .get_or_compile_program(&kernel_key, || repeak_source(&type_name, &type_suffix))?;

      let kernel = Kernel::builder()
        .program(&program)
        .name(kernel_key)
        .queue(self.env.queue.clone())
        .global_work_size(self.shape)
        .arg(&self.buffer)
        .arg(self.strides[0] as i32)
        .arg(self.strides[1] as i32)
        .arg(self.offset[0] as i32)
        .arg(self.offset[1] as i32)
        .arg(&output_buffer)
        .arg(self.rows())
        .arg(self.cols())
        .build()?;

      unsafe {
        kernel.enq()?;
      }

      Ok(GPUMatrix {
        buffer: output_buffer,
        shape: self.shape.clone(),
        strides: [self.cols(), 1],
        offset: self.offset.clone(),
        env: self.env.clone(),
      })
    }
  }

  pub fn to_cpu(&self) -> Result<Matrix<T>, Error> {
    let contiguous_self = self.contiguous()?;
    let mut data = vec![T::default(); contiguous_self.buffer.len()];
    contiguous_self
      .buffer
      .read(&mut data)
      .enq()
      .map_err(|e| Error::OclError(e))?;
    let start = contiguous_self.offset[0] * contiguous_self.strides[0]
      + contiguous_self.offset[1] * contiguous_self.strides[1];
    let end = start + contiguous_self.len() - 1;
    let data = data[start..=end].to_vec();
    Matrix::from_vec(contiguous_self.shape.clone(), data)
  }

  pub fn write(&self, data: &GPUMatrix<T>) -> Result<(), Error> {
    if data.shape != self.shape {
      return Err(Error::MismatchedShape {
        expected: vec![self.shape[0], self.shape[1]],
        found: vec![data.shape[0], data.shape[1]],
      });
    }

    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("write_mat_{}", type_suffix);

    let program = self.env.get_or_compile_program(&kernel_name, || {
      crate::clarray::kernel::matrix::write_source(&type_name, &type_suffix)
    })?;

    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size(self.shape)
      .arg(&data.buffer)
      .arg(data.strides[0] as i32)
      .arg(data.strides[1] as i32)
      .arg(data.offset[0] as i32)
      .arg(data.offset[1] as i32)
      .arg(&self.buffer)
      .arg(self.strides[0] as i32)
      .arg(self.strides[1] as i32)
      .arg(self.offset[0] as i32)
      .arg(self.offset[1] as i32)
      .arg(self.rows() as i32)
      .arg(self.cols() as i32)
      .build()?;

    unsafe {
      kernel.enq()?;
    }

    Ok(())
  }

  pub fn row_iter(&self) -> impl Iterator<Item = GPUVector<T>> + '_ {
    let contiguous_self = self.contiguous().unwrap();
    (0..contiguous_self.rows()).map(move |i| GPUVector {
      buffer: contiguous_self.buffer.clone(),
      shape: [contiguous_self.cols()],
      strides: [contiguous_self.strides[1]],
      offset: [i * contiguous_self.strides[0]],
      env: self.env.clone(),
    })
  }

  pub fn row_sum(&self) -> Result<GPUVector<T>, Error> {
    let contiguous_self = self.contiguous()?;
    let output = GPUVector::zeros([contiguous_self.rows()], self.env.clone())?;

    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("row_sum_mat_{}", type_suffix);

    let program = self
      .env
      .get_or_compile_program(&kernel_name, || row_sum_source(&type_name, &type_suffix))?;

    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size([contiguous_self.rows()])
      .arg(&contiguous_self.buffer)
      .arg(contiguous_self.strides[0] as i32)
      .arg(contiguous_self.strides[1] as i32)
      .arg(contiguous_self.offset[0] as i32)
      .arg(contiguous_self.offset[1] as i32)
      .arg(&output.buffer)
      .arg(output.strides[0] as i32)
      .arg(output.offset[0] as i32)
      .arg(contiguous_self.cols() as i32)
      .build()?;

    unsafe {
      kernel.enq()?;
    }

    Ok(output)
  }

  pub fn from_diag(vec: &GPUVector<T>, env: Arc<GPUEnv>) -> Result<Self, Error> {
    let output = GPUMatrix::zeros([vec.shape[0], vec.shape[0]], env.clone())?;
    let type_suffix = std::any::type_name::<T>();
    let type_name = clang_type_name(&type_suffix);
    let kernel_name = format!("diag_mat_{}", type_suffix);
    let program = env.get_or_compile_program(&kernel_name, || {
      crate::clarray::kernel::matrix::diag_source(&type_name, &type_suffix)
    })?;
    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(env.queue.clone())
      .global_work_size(vec.shape[0])
      .arg(&vec.buffer)
      .arg(vec.strides[0] as i32)
      .arg(vec.offset[0] as i32)
      .arg(&output.buffer)
      .arg(output.strides[0] as i32)
      .arg(output.strides[1] as i32)
      .arg(output.offset[0] as i32)
      .arg(output.offset[1] as i32)
      .arg(vec.shape[0] as i32)
      .arg(vec.shape[0] as i32)
      .build()?;
    unsafe {
      kernel.enq()?;
    }
    Ok(output)
  }
}
