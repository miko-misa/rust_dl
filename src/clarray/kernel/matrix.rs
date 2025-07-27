use crate::clarray::kernel::op_to_suffix;

pub fn repeak_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void repeak_mat_{type_suffix}(
      __global const {type_name}* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global {type_name}* output,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;


      unsigned long input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
      unsigned long output_index = row * cols + col;

      output[output_index] = input[input_index];

    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn elementwise_op_source(op: &str, type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void elementwise_{op_name}_mat_{type_suffix}(
      __global const {type_name}* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const {type_name}* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global {type_name}* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;


      c[c_index] = a[a_index] {op} b[b_index];
    }}
  "#,
    op_name = op_to_suffix(op),
    op = op,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn elementwise_op_scalar_r_source(op: &str, type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void elementwise_{op_name}_scalar_r_mat_{type_suffix}(
      __global const {type_name}* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      {type_name} b,
      __global {type_name}* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      // print a memory length

      c[c_index] = a[a_index] {op} b;
    }}
  "#,
    op_name = op_to_suffix(op),
    op = op,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn elementwise_op_scalar_l_source(op: &str, type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void elementwise_{op_name}_scalar_l_mat_{type_suffix}(
      {type_name} a,
      __global const {type_name}* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global {type_name}* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      c[c_index] = a {op} b[b_index];
    }}
  "#,
    op_name = op_to_suffix(op),
    op = op,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn dot_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void dot_mat_{type_suffix}(
      __global const {type_name}* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const {type_name}* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global {type_name}* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long M, unsigned long N, unsigned long K
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= M || col >= N) return;

      {type_name} sum = 0.0;
      for (unsigned long k = 0; k < K; ++k) {{
        unsigned long a_index = (row + a_offset_0) * a_stride_0 + (k + a_offset_1) * a_stride_1;
        unsigned long b_index = (k + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
        sum += a[a_index] * b[b_index];
      }}

      c[(row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1] = sum;
    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn write_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void write_mat_{type_suffix}(
      __global {type_name}* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global {type_name}* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
      unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;
      output[output_index] = input[input_index];
    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn clip_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
      __kernel void clip_mat_{type_suffix}(
        __global const {type_name}* input,
        unsigned long input_stride_0, unsigned long input_stride_1,
        unsigned long input_offset_0, unsigned long input_offset_1,
        __global {type_name}* output,
        unsigned long output_stride_0, unsigned long output_stride_1,
        unsigned long output_offset_0, unsigned long output_offset_1,
        {type_name} min_val, {type_name} max_val,
        unsigned long rows, unsigned long cols
      ) {{

        unsigned long row = get_global_id(0);
        unsigned long col = get_global_id(1);

        if (row >= rows || col >= cols) return;

        unsigned long input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;

        {type_name} value = input[input_index];
        if (value < min_val) {{
          value = min_val;
        }} else if (value > max_val) {{
          value = max_val;
        }}
        output[output_index] = value;
      }}
    "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn row_sum_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void row_sum_mat_{type_suffix}(
      __global const {type_name}* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global {type_name}* output,
      unsigned long output_stride,
      unsigned long output_offset,
      unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);

      {type_name} sum = 0.0;
      for (unsigned long col = 0; col < cols; ++col) {{
        unsigned long index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        sum += input[index];
      }}

      output[(row + output_offset) * output_stride] = sum;
    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn diag_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void diag_mat_{type_suffix}(
      __global const {type_name}* input,
      unsigned long input_stride,
      unsigned long input_offset,
      __global {type_name}* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);
      if (row >= rows || col >= cols) return;

      unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;

      if (row == col) {{
        unsigned long input_index = (row + input_offset) * input_stride;
        output[output_index] = input[input_index];
      }} else {{
        output[output_index] = 0.0;
      }}
    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}

pub fn row_max_mask_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void row_max_mask_mat_{type_suffix}(
      __global const {type_name}* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global {type_name}* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long rows, unsigned long cols
    ) {{

      unsigned long row = get_global_id(0);
      if (row >= rows) return;

      {type_name} max_val = -1e10;
      unsigned long max_col = -1;
      for (unsigned long col = 0; col < cols; ++col) {{
        unsigned long index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        {type_name} value = input[index];
        if (value > max_val) {{
          max_val = value;
          max_col = col;
        }}
      }}
      output[(row + output_offset_0) * output_stride_0 + (max_col + output_offset_1) * output_stride_1] = 1.0;
    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}
