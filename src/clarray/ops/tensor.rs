use ocl::{Kernel, builders::BufferBuilder};
use std::fmt::Debug;

use crate::clarray::{
  error::Error,
  kernel::op_to_suffix,
  tensor::{GPUMatrix, GPUTensor, OclComputeNum},
};

impl<T> GPUTensor<T, [usize; 4]>
where
  T: OclComputeNum,
{
  pub fn im2col(&self, filter_size: [usize; 3], stride: [usize; 3]) -> Result<GPUMatrix<T>, Error> {
    let [batch_size, channels, height, width] = self.shape;
    let channel_size = (channels - filter_size[0]) / stride[0] + 1;
    let col_size = (height - filter_size[1]) / stride[1] + 1;
    let row_size = (width - filter_size[2]) / stride[2] + 1;
    let output = GPUMatrix::<T>::zeros(
      [
        channel_size * col_size * row_size * batch_size,
        filter_size[0] * filter_size[1] * filter_size[2],
      ],
      self.env.clone(),
    )?;

    let type_suffix = std::any::type_name::<T>();
    let kernel_name = format!("im2col_{}", type_suffix);
    let program = self.env.get_or_compile_program(&type_suffix)?;
    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size([
        batch_size,
        col_size * row_size * channel_size,
        filter_size[0] * filter_size[1] * filter_size[2],
      ])
      .arg(&self.buffer)
      .arg(self.strides[0] as u64)
      .arg(self.strides[1] as u64)
      .arg(self.strides[2] as u64)
      .arg(self.strides[3] as u64)
      .arg(self.offset[0] as u64)
      .arg(self.offset[1] as u64)
      .arg(self.offset[2] as u64)
      .arg(self.offset[3] as u64)
      .arg(&output.buffer)
      .arg(output.strides[0] as u64)
      .arg(output.strides[1] as u64)
      .arg(output.offset[0] as u64)
      .arg(output.offset[1] as u64)
      .arg((filter_size[1] * filter_size[2]) as u64)
      .arg(filter_size[2] as u64)
      .arg(1 as u64)
      .arg(channel_size as u64)
      .arg(col_size as u64)
      .arg(row_size as u64)
      .arg(stride[0] as u64)
      .arg(stride[1] as u64)
      .arg(stride[2] as u64)
      .build()?;

    unsafe {
      kernel.enq()?;
    }

    Ok(output)
  }

  pub fn padding(
    &self,
    c_pad: [i32; 2],
    h_pad: [i32; 2],
    w_pad: [i32; 2],
  ) -> Result<GPUTensor<T, [usize; 4]>, Error> {
    let [pad_front, pad_back] = c_pad;
    let [pad_top, pad_bottom] = h_pad;
    let [pad_left, pad_right] = w_pad;
    let [N, C, H, W] = self.shape;
    let out_c = (C as i32 + pad_front + pad_back) as usize;
    let out_w = (W as i32 + pad_left + pad_right) as usize;
    let out_h = (H as i32 + pad_top + pad_bottom) as usize;
    let output_shape = [N, out_c, out_h, out_w];
    let output = GPUTensor::<T, [usize; 4]>::zeros(output_shape, self.env.clone())?;

    let type_suffix = std::any::type_name::<T>();
    let kernel_name = format!("padding_{}", type_suffix);
    let program = self.env.get_or_compile_program(&type_suffix)?;
    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size([N * out_c, out_h, out_w])
      .arg(&self.buffer)
      .arg(N as u64)
      .arg(C as u64)
      .arg(H as u64)
      .arg(W as u64)
      .arg(self.strides[0] as u64)
      .arg(self.strides[1] as u64)
      .arg(self.strides[2] as u64)
      .arg(self.strides[3] as u64)
      .arg(self.offset[0] as u64)
      .arg(self.offset[1] as u64)
      .arg(self.offset[2] as u64)
      .arg(self.offset[3] as u64)
      .arg(&output.buffer)
      .arg(out_c as u64)
      .arg(out_h as u64)
      .arg(out_w as u64)
      .arg(output.strides[0] as u64)
      .arg(output.strides[1] as u64)
      .arg(output.strides[2] as u64)
      .arg(output.strides[3] as u64)
      .arg(output.offset[0] as u64)
      .arg(output.offset[1] as u64)
      .arg(output.offset[2] as u64)
      .arg(output.offset[3] as u64)
      .arg(pad_front)
      .arg(pad_back)
      .arg(pad_top)
      .arg(pad_bottom)
      .arg(pad_left)
      .arg(pad_right)
      .build()?;
    unsafe {
      kernel.enq()?;
    }
    Ok(output)
  }
}

fn elementwise_op<T, D>(
  op: &str,
  input1: &GPUTensor<T, D>,
  input2: &GPUTensor<T, D>,
) -> Result<GPUTensor<T, D>, Error>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  let type_suffix = std::any::type_name::<T>();
  let kernel_name = format!("elementwise_{}_tensor_{}", op_to_suffix(op), type_suffix);
  let program = input1.env.get_or_compile_program(&type_suffix)?;
  let output = GPUTensor::<T, D>::zeros(input1.shape.clone(), input1.env.clone())?;

  // strideを配列へ
  let input1_strides = input1.strides.as_ref();
  let input2_strides = input2.strides.as_ref();
  let output_strides = output.strides.as_ref();

  let dim = input1_strides.len() as i32;

  let kernel = Kernel::builder()
    .program(&program)
    .name(&kernel_name)
    .queue(input1.env.queue.clone())
    .global_work_size(input1.len())
    .arg(&input1.buffer)
    .arg(
      BufferBuilder::new()
        .len(dim)
        .queue(input1.env.queue.clone())
        .copy_host_slice(input1_strides)
        .build()?,
    )
    .arg(&input2.buffer)
    .arg(
      BufferBuilder::new()
        .len(dim)
        .queue(input1.env.queue.clone())
        .copy_host_slice(input2_strides)
        .build()?,
    )
    .arg(&output.buffer)
    .arg(
      BufferBuilder::new()
        .len(dim)
        .queue(input1.env.queue.clone())
        .copy_host_slice(output_strides)
        .build()?,
    )
    .arg(dim as u64)
    .build()?;

  unsafe {
    kernel.enq()?;
  }

  Ok(output)
}

macro_rules! impl_tensor_elementwise_op {
  ($trait:ident, $method:ident, $op_symbol:expr) => {
    impl<'a, 'b, T, D> std::ops::$trait<&'b GPUTensor<T, D>> for &'a GPUTensor<T, D>
    where
      T: OclComputeNum,
      D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
    {
      type Output = Result<GPUTensor<T, D>, Error>;
      fn $method(self, rhs: &'b GPUTensor<T, D>) -> Self::Output {
        Ok(elementwise_op($op_symbol, self, rhs)?)
      }
    }
  };
}

impl_tensor_elementwise_op!(Add, add, "+");
impl_tensor_elementwise_op!(Sub, sub, "-");
impl_tensor_elementwise_op!(Mul, mul, "*");
impl_tensor_elementwise_op!(Div, div, "/");

impl<T, D> GPUTensor<T, D>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  pub fn relu_mask(&self) -> Result<GPUTensor<T, D>, Error> {
    let type_suffix = std::any::type_name::<T>();
    let kernel_name = format!("relu_mask_tensor_{}", type_suffix);
    let program = self.env.get_or_compile_program(&type_suffix)?;
    let output = GPUTensor::<T, D>::zeros(self.shape.clone(), self.env.clone())?;

    let input_strides = self.strides.as_ref();
    let output_strides = output.strides.as_ref();

    let dim = input_strides.len() as i32;

    let kernel = Kernel::builder()
      .program(&program)
      .name(&kernel_name)
      .queue(self.env.queue.clone())
      .global_work_size(self.len())
      .arg(&self.buffer)
      .arg(
        BufferBuilder::new()
          .len(dim)
          .queue(self.env.queue.clone())
          .copy_host_slice(input_strides)
          .build()?,
      )
      .arg(&output.buffer)
      .arg(
        BufferBuilder::new()
          .len(dim)
          .queue(self.env.queue.clone())
          .copy_host_slice(output_strides)
          .build()?,
      )
      .arg(dim as u64)
      .build()?;

    unsafe {
      kernel.enq()?;
    }

    Ok(output)
  }
}
