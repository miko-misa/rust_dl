use crate::clarray::kernel::op_to_suffix;

pub fn repeak_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void repeak_mat_{type_suffix}(
      __global const {type_name}* input,
      int input_stride_0, int input_stride_1,
      int input_offset_0, int input_offset_1,
      __global {type_name}* output,
      int rows, int cols
    ) {{

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row >= rows || col >= cols) return;


      int input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
      int output_index = row * cols + col;

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
      int a_stride_0, int a_stride_1,
      int a_offset_0, int a_offset_1,
      __global const {type_name}* b,
      int b_stride_0, int b_stride_1,
      int b_offset_0, int b_offset_1,
      __global {type_name}* c,
      int c_stride_0, int c_stride_1,
      int c_offset_0, int c_offset_1,
      int rows, int cols
    ) {{

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      int a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      int b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      int c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;


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
      int a_stride_0, int a_stride_1,
      int a_offset_0, int a_offset_1,
      {type_name} b,
      __global {type_name}* c,
      int c_stride_0, int c_stride_1,
      int c_offset_0, int c_offset_1,
      int rows, int cols
    ) {{

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      int a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      int c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

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
      int b_stride_0, int b_stride_1,
      int b_offset_0, int b_offset_1,
      __global {type_name}* c,
      int c_stride_0, int c_stride_1,
      int c_offset_0, int c_offset_1,
      int rows, int cols
    ) {{

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      int b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      int c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

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
      int a_stride_0, int a_stride_1,
      int a_offset_0, int a_offset_1,
      __global const {type_name}* b,
      int b_stride_0, int b_stride_1,
      int b_offset_0, int b_offset_1,
      __global {type_name}* c,
      int c_stride_0, int c_stride_1,
      int c_offset_0, int c_offset_1,
      int M, int N, int K
    ) {{

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row >= M || col >= N) return;

      {type_name} sum = 0.0;
      for (int k = 0; k < K; ++k) {{
        int a_index = (row + a_offset_0) * a_stride_0 + (k + a_offset_1) * a_stride_1;
        int b_index = (k + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
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
      int input_stride_0, int input_stride_1,
      int input_offset_0, int input_offset_1,
      __global {type_name}* output,
      int output_stride_0, int output_stride_1,
      int output_offset_0, int output_offset_1,
      int rows, int cols
    ) {{

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      int input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
      int output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;
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
        int input_stride_0, int input_stride_1,
        int input_offset_0, int input_offset_1,
        __global {type_name}* output,
        int output_stride_0, int output_stride_1,
        int output_offset_0, int output_offset_1,
        {type_name} min_val, {type_name} max_val,
        int rows, int cols
      ) {{

        int row = get_global_id(0);
        int col = get_global_id(1);

        if (row >= rows || col >= cols) return;

        int input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        int output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;

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
      int input_stride_0, int input_stride_1,
      int input_offset_0, int input_offset_1,
      __global {type_name}* output,
      int output_stride,
      int output_offset,
      int rows,
    ) {{

      int row = get_global_id(0);
      if (row >= rows) return;

      {type_name} sum = 0.0;
      for (int col = 0; col < cols; ++col) {{
        int index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
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
      int input_stride,
      int input_offset,
      __global {type_name}* output,
      int output_stride_0, int output_stride_1,
      int output_offset_0, int output_offset_1,
      int rows, int cols
    ) {{

      int row = get_global_id(0);
      if (row >= rows) return;

      int output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;

      for (int col = 0; col < cols; ++col) {{
        if (row == col) {{
          int input_index = (row + input_offset) * input_stride;
          output[output_index] = input[input_index];
        }} else {{
          output[output_index] = 0.0;
        }}
      }}
    }}
  "#,
    type_suffix = type_suffix,
    type_name = type_name
  )
}
