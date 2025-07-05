use std::{fmt::Debug, sync::Arc};

use num_traits::Num;
use ocl::{Buffer, Device, Kernel, OclPrm, core::OclNum, flags};

use crate::clarray::{
  env::{GPUEnv, env},
  error::Error,
  kernel::{clang_type_name, matrix::repeak_source},
};

pub struct Tensor<T, D>
where
  T: OclPrm + Num + Copy + Debug + Default + OclNum,
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
  T: OclPrm + Num + Copy + Debug + Default + OclNum,
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
  T: OclPrm + Num + Copy + Debug + Default + OclNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  pub(crate) buffer: Buffer<T>,
  pub shape: D,
  pub strides: D,
  pub offset: D,
  pub(crate) env: Arc<GPUEnv>,
}

pub type GPUVector<T> = GPUTensor<T, [usize; 1]>;
pub type GPUMatrix<T> = GPUTensor<T, [usize; 2]>;

impl<T> GPUVector<T>
where
  T: OclPrm + Num + Copy + Debug + Default + OclNum,
{
  pub fn new(shape: [usize; 1], env: Arc<GPUEnv>) -> Result<Self, Error> {
    if shape.len() != 1 {
      return Err(Error::MismatchedShape {
        expected: vec![1],
        found: vec![shape.len()],
      });
    }

    let buffer = Buffer::<T>::builder()
      .queue(env.queue.clone())
      .len(shape[0])
      .build()?;

    Ok(GPUVector {
      buffer,
      shape,
      strides: [1],
      offset: [0],
      env,
    })
  }

  pub fn len(&self) -> usize {
    self.shape[0]
  }

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
  T: OclPrm + Num + Copy + Debug + Default + OclNum,
{
  pub fn new(shape: [usize; 2], env: Arc<GPUEnv>) -> Result<Self, Error> {
    if shape.len() != 2 {
      return Err(Error::MismatchedShape {
        expected: vec![2],
        found: vec![shape.len()],
      });
    }

    let buffer = Buffer::<T>::builder()
      .queue(env.queue.clone())
      .len(shape[0] * shape[1])
      .build()?;

    Ok(GPUMatrix {
      buffer,
      shape,
      strides: [shape[1], 1],
      offset: [0, 0],
      env,
    })
  }
  pub fn rows(&self) -> usize {
    self.shape[0]
  }

  pub fn cols(&self) -> usize {
    self.shape[1]
  }

  pub fn len(&self) -> usize {
    self.shape.iter().product()
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
}
