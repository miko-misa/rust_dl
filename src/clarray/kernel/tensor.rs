use crate::clarray::kernel::op_to_suffix;

pub fn elementwise_op_source(op: &str, type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void elementwise_{op_name}_tensor_{type_suffix}(
      __global const {type_name}* input1,
      __global const size_t* input1_strides,
      __global const {type_name}* input2,
      __global const size_t* input2_strides,
      __global {type_name}* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {{
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {{0}};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {{
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }}
      unsigned long input1_index = 0;
      unsigned long input2_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {{
        input1_index += idx[i] * input1_strides[i];
        input2_index += idx[i] * input2_strides[i];
        output_index += idx[i] * output_strides[i];
      }}

      output[output_index] = input1[input1_index] {op} input2[input2_index];
    }}
    "#,
    op = op,
    op_name = op_to_suffix(op),
    type_name = type_name,
    type_suffix = type_suffix
  )
}

pub fn im2col_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void im2col_{type_suffix}(
      __global const {type_name}* input,
      unsigned long input_stride_0, unsigned long input_stride_1, unsigned long input_stride_2, unsigned long input_stride_3,
      unsigned long input_offset_0, unsigned long input_offset_1, unsigned long input_offset_2, unsigned long input_offset_3,
      __global {type_name}* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long filter_stride_0, unsigned long filter_stride_1, unsigned long filter_stride_2,
      unsigned long channel_size, unsigned long col_size, unsigned long row_size,
      unsigned long stride_0, unsigned long stride_1, unsigned long stride_2
    ) {{
      unsigned long batch = get_global_id(0);
      unsigned long row = get_global_id(1);
      unsigned long col = get_global_id(2);

      unsigned long output_row = channel_size * col_size * row_size * batch + row;
      unsigned long output_col = col;

      unsigned long k = row / (row_size * col_size);
      unsigned long i = (row - (k * row_size * col_size)) / row_size;
      unsigned long j = row % row_size;

      unsigned long filter_start_C = k * stride_0;
      unsigned long filter_start_H = i * stride_1;
      unsigned long filter_start_W = j * stride_2;

      unsigned long filter_offset_C = col / filter_stride_0;
      unsigned long filter_offset_H = (col % filter_stride_0) / filter_stride_1;
      unsigned long filter_offset_W = ((col % filter_stride_0) % filter_stride_1) / filter_stride_2;

      unsigned long input_index = (
        (batch + input_offset_0) * input_stride_0 +
        (filter_start_C + filter_offset_C + input_offset_1) * input_stride_1 +
        (filter_start_H + filter_offset_H + input_offset_2) * input_stride_2 +
        (filter_start_W + filter_offset_W + input_offset_3) * input_stride_3
      );
      unsigned long output_index = (
        (output_row + output_offset_0) * output_stride_0 +
        (output_col + output_offset_1) * output_stride_1
      );

      output[output_index] = input[input_index];
    }}
    "#,
    type_name = type_name,
    type_suffix = type_suffix
  )
}

pub fn col2im_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    #pragma OPENCL EXTENSION cl_khr_unsigned long64_base_atomics : enable
    #pragma OPENCL EXTENSION cl_khr_unsigned long64_extended_atomics : enable
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    void atomic_add_double(volatile global double *addr, double val) {{
      union {{
        double d;
        ulong ul;
      }} old_val, new_val;
      old_val.ul = *((volatile global ulong*)addr);
      do {{
        new_val.d = old_val.d + val;
        ulong read_val = atom_cmpxchg((volatile global ulong*)addr, old_val.ul, new_val.ul);
        if (read_val == old_val.ul) {{
            break;
        }}
        old_val.ul = read_val;
      }} while (true);
    }}
    __kernel void col2im_{type_suffix}(
      __global const {type_name}* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global {type_name}* output,
      unsigned long output_stride_0, unsigned long output_stride_1, unsigned long output_stride_2, unsigned long output_stride_3,
      unsigned long output_offset_0, unsigned long output_offset_1, unsigned long output_offset_2, unsigned long output_offset_3,
      unsigned long filter_stride_0, unsigned long filter_stride_1, unsigned long filter_stride_2,
      unsigned long channel_size, unsigned long col_size, unsigned long row_size,
      unsigned long stride_0, unsigned long stride_1, unsigned long stride_2
    ) {{
      unsigned long batch = get_global_id(0);
      unsigned long row = get_global_id(1);
      unsigned long col = get_global_id(2);

      unsigned long input_row = channel_size * col_size * row_size * batch + row;
      unsigned long input_col = col;

      unsigned long k = row / (row_size * col_size);
      unsigned long i = (row - (k * row_size * col_size)) / row_size;
      unsigned long j = row % row_size;

      unsigned long filter_start_C = k * stride_0;
      unsigned long filter_start_H = i * stride_1;
      unsigned long filter_start_W = j * stride_2;

      unsigned long filter_offset_C = col / filter_stride_0;
      unsigned long filter_offset_H = (col % filter_stride_0) / filter_stride_1;
      unsigned long filter_offset_W = ((col % filter_stride_0) % filter_stride_1) / filter_stride_2;

      unsigned long output_index = (
        (batch + output_offset_0) * output_stride_0 +
        (filter_start_C + filter_offset_C + output_offset_1) * output_stride_1 +
        (filter_start_H + filter_offset_H + output_offset_2) * output_stride_2 +
        (filter_start_W + filter_offset_W + output_offset_3) * output_stride_3
      );
      unsigned long input_index = (
        (input_row + input_offset_0) * input_stride_0 +
        (input_col + input_offset_1) * input_stride_1
      );
      atomic_add_{type_name}(&output[output_index], input[input_index]);
    }}
    "#,
    type_name = type_name,
    type_suffix = type_suffix
  )
}

pub fn padding_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void padding_{type_suffix}(
      __global const {type_name}* input,
      unsigned long N, unsigned long C, unsigned long H, unsigned long W,
      unsigned long input_stride_0, unsigned long input_stride_1, unsigned long input_stride_2, unsigned long input_stride_3,
      unsigned long input_offset_0, unsigned long input_offset_1, unsigned long input_offset_2, unsigned long input_offset_3,
      __global {type_name}* output,
      unsigned long C_out, unsigned long H_out, unsigned long W_out,
      unsigned long output_stride_0, unsigned long output_stride_1, unsigned long output_stride_2, unsigned long output_stride_3,
      unsigned long output_offset_0, unsigned long output_offset_1, unsigned long output_offset_2, unsigned long output_offset_3,
      int pad_front_c, int pad_back_c,
      int pad_top, int pad_bottom, int pad_left, int pad_right
    ) {{
      unsigned long gid0 = get_global_id(0);
      unsigned long x = get_global_id(2);
      unsigned long y = get_global_id(1);
      if (x >= W_out || y >= H_out) return;

      unsigned long nc = gid0;
      unsigned long n = nc / C_out;
      unsigned long c_out = nc % C_out;
      if (n >= N) return;

      unsigned long c_in = c_out - pad_front_c;
      unsigned long in_y = y - pad_top;
      unsigned long in_x = x - pad_left;

      {type_name} v = ({type_name})0;

      if (c_in >= 0 && c_in < C
          && in_y >= 0 && in_y < H
          && in_x >= 0 && in_x < W) {{
        unsigned long ii = input_offset_0 + n*input_stride_0
               + input_offset_1 + c_in*input_stride_1
               + input_offset_2 + in_y*input_stride_2
               + input_offset_3 + in_x*input_stride_3;
        v = input[ii];
      }}

      unsigned long oo = output_offset_0 + n*output_stride_0
             + output_offset_1 + c_out*output_stride_1
             + output_offset_2 + y*output_stride_2
             + output_offset_3 + x*output_stride_3;
      output[oo] = v;
    }}
    "#,
    type_name = type_name,
    type_suffix = type_suffix
  )
}

pub fn relu_mask_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void relu_mask_tensor_{type_suffix}(
      __global const {type_name}* input,
      __global const size_t* input_strides,
      __global {type_name}* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {{
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {{0}};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {{
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }}
      unsigned long input_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {{
        input_index += idx[i] * input_strides[i];
        output_index += idx[i] * output_strides[i];
      }}

      output[output_index] = (input[input_index] > 0) ? ({type_name})1 : ({type_name})0;
    }}
    "#,
    type_name = type_name,
    type_suffix = type_suffix
  )
}
