use crate::clarray::kernel::op_to_suffix;

pub fn broadcast_matrix_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
        __kernel void broadcast_matrix_vec_{type_suffix}(
            __global const {type_name}* input,
            unsigned long input_stride,
            unsigned long input_offset,
            __global {type_name}* output,
            unsigned long output_stride_0, unsigned long output_stride_1,
            unsigned long output_offset_0, unsigned long output_offset_1,
            const unsigned long rows,
            const unsigned long cols
        ) {{
            unsigned long row = get_global_id(0);
            unsigned long col = get_global_id(1);
            if (row >= rows || col >= cols) return;
            unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;
            output[output_index] = input[(col + input_offset) * input_stride];
        }}
        "#,
    type_name = type_name,
    type_suffix = type_suffix
  )
}

pub fn sum_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
        __kernel void sum_vec_{type_suffix}(
            __global const {type_name}* input,
            unsigned long input_stride,
            unsigned long input_offset,
            __global {type_name}* output,
            const unsigned long len
        ) {{
            {type_name} sum = 0;
            for (unsigned long col = 0; col < len; ++col) {{
                sum += input[(col + input_offset) * input_stride];
            }}
            output[0] = sum;
        }}
        "#,
    type_name = type_name,
    type_suffix = type_suffix
  )
}

pub fn write_source(type_name: &str, type_suffix: &str) -> String {
  format!(
    r#"
    __kernel void write_vec_{type_suffix}(
      __global {type_name}* input,
      unsigned long input_stride,
      unsigned long input_offset,
      __global {type_name}* output,
      unsigned long output_stride,
      unsigned long output_offset,
      unsigned long len
    ) {{
      unsigned long index = get_global_id(0);
      if (index >= len) return;
      unsigned long input_index = (index + input_offset) * input_stride;
      unsigned long output_index = (index + output_offset) * output_stride;
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
    __kernel void elementwise_{op_name}_vec_{type_suffix}(
      __global const {type_name}* a,
      unsigned long a_stride,
      unsigned long a_offset,
      __global const {type_name}* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global {type_name}* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {{

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

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
    __kernel void elementwise_{op_name}_scalar_r_vec_{type_suffix}(
      __global const {type_name}* a,
      unsigned long a_stride,
      unsigned long a_offset,
      {type_name} b,
      __global {type_name}* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {{

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

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
    __kernel void elementwise_{op_name}_scalar_l_vec_{type_suffix}(
      {type_name} a,
      __global const {type_name}* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global {type_name}* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {{

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a {op} b[b_index];
    }}
  "#,
    op_name = op_to_suffix(op),
    op = op,
    type_suffix = type_suffix,
    type_name = type_name
  )
}
