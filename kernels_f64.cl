
// Kernels from vector.rs (type: double)

        __kernel void broadcast_matrix_vec_f64(
            __global const double* input,
            unsigned long input_stride,
            unsigned long input_offset,
            __global double* output,
            unsigned long output_stride_0, unsigned long output_stride_1,
            unsigned long output_offset_0, unsigned long output_offset_1,
            const unsigned long rows,
            const unsigned long cols
        ) {
            unsigned long row = get_global_id(0);
            unsigned long col = get_global_id(1);
            if (row >= rows || col >= cols) return;
            unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;
            output[output_index] = input[(col + input_offset) * input_stride];
        }
        
        __kernel void sum_vec_f64(
            __global const double* input,
            unsigned long input_stride,
            unsigned long input_offset,
            __global double* output,
            const unsigned long len
        ) {
            double sum = 0;
            for (unsigned long col = 0; col < len; ++col) {
                sum += input[(col + input_offset) * input_stride];
            }
            output[0] = sum;
        }
        
    __kernel void write_vec_f64(
      __global double* input,
      unsigned long input_stride,
      unsigned long input_offset,
      __global double* output,
      unsigned long output_stride,
      unsigned long output_offset,
      unsigned long len
    ) {
      unsigned long index = get_global_id(0);
      if (index >= len) return;
      unsigned long input_index = (index + input_offset) * input_stride;
      unsigned long output_index = (index + output_offset) * output_stride;
      output[output_index] = input[input_index];
    }
  
    __kernel void elementwise_add_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] + b[b_index];
    }
  
    __kernel void elementwise_add_scalar_r_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      double b,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] + b;
    }
  
    __kernel void elementwise_add_scalar_l_vec_f64(
      double a,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a + b[b_index];
    }
  
    __kernel void elementwise_sub_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] - b[b_index];
    }
  
    __kernel void elementwise_sub_scalar_r_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      double b,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] - b;
    }
  
    __kernel void elementwise_sub_scalar_l_vec_f64(
      double a,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a - b[b_index];
    }
  
    __kernel void elementwise_mul_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] * b[b_index];
    }
  
    __kernel void elementwise_mul_scalar_r_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      double b,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] * b;
    }
  
    __kernel void elementwise_mul_scalar_l_vec_f64(
      double a,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a * b[b_index];
    }
  
    __kernel void elementwise_div_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] / b[b_index];
    }
  
    __kernel void elementwise_div_scalar_r_vec_f64(
      __global const double* a,
      unsigned long a_stride,
      unsigned long a_offset,
      double b,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long a_index = (index + a_offset) * a_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a[a_index] / b;
    }
  
    __kernel void elementwise_div_scalar_l_vec_f64(
      double a,
      __global const double* b,
      unsigned long b_stride,
      unsigned long b_offset,
      __global double* c,
      unsigned long c_stride,
      unsigned long c_offset,
      unsigned long len
    ) {

      unsigned long index = get_global_id(0);

      if (index >= len) return;

      unsigned long b_index = (index + b_offset) * b_stride;
      unsigned long c_index = (index + c_offset) * c_stride;

      c[c_index] = a / b[b_index];
    }
  
// Kernels from matrix.rs (type: double)

    __kernel void repeak_mat_f64(
      __global const double* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global double* output,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;


      unsigned long input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
      unsigned long output_index = row * cols + col;

      output[output_index] = input[input_index];

    }
  
    __kernel void dot_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long M, unsigned long N, unsigned long K
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= M || col >= N) return;

      double sum = 0.0;
      for (unsigned long k = 0; k < K; ++k) {
        unsigned long a_index = (row + a_offset_0) * a_stride_0 + (k + a_offset_1) * a_stride_1;
        unsigned long b_index = (k + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
        sum += a[a_index] * b[b_index];
      }

      c[(row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1] = sum;
    }
  
    __kernel void write_mat_f64(
      __global double* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global double* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
      unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;
      output[output_index] = input[input_index];
    }
  
      __kernel void clip_mat_f64(
        __global const double* input,
        unsigned long input_stride_0, unsigned long input_stride_1,
        unsigned long input_offset_0, unsigned long input_offset_1,
        __global double* output,
        unsigned long output_stride_0, unsigned long output_stride_1,
        unsigned long output_offset_0, unsigned long output_offset_1,
        double min_val, double max_val,
        unsigned long rows, unsigned long cols
      ) {

        unsigned long row = get_global_id(0);
        unsigned long col = get_global_id(1);

        if (row >= rows || col >= cols) return;

        unsigned long input_index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;

        double value = input[input_index];
        if (value < min_val) {
          value = min_val;
        } else if (value > max_val) {
          value = max_val;
        }
        output[output_index] = value;
      }
    
    __kernel void row_sum_mat_f64(
      __global const double* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global double* output,
      unsigned long output_stride,
      unsigned long output_offset,
      unsigned long cols
    ) {

      unsigned long row = get_global_id(0);

      double sum = 0.0;
      for (unsigned long col = 0; col < cols; ++col) {
        unsigned long index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        sum += input[index];
      }

      output[(row + output_offset) * output_stride] = sum;
    }
  
    __kernel void diag_mat_f64(
      __global const double* input,
      unsigned long input_stride,
      unsigned long input_offset,
      __global double* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);
      if (row >= rows || col >= cols) return;

      unsigned long output_index = (row + output_offset_0) * output_stride_0 + (col + output_offset_1) * output_stride_1;

      if (row == col) {
        unsigned long input_index = (row + input_offset) * input_stride;
        output[output_index] = input[input_index];
      } else {
        output[output_index] = 0.0;
      }
    }
  
    __kernel void row_max_mask_mat_f64(
      __global const double* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global double* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      if (row >= rows) return;

      double max_val = -1e10;
      unsigned long max_col = -1;
      for (unsigned long col = 0; col < cols; ++col) {
        unsigned long index = (row + input_offset_0) * input_stride_0 + (col + input_offset_1) * input_stride_1;
        double value = input[index];
        if (value > max_val) {
          max_val = value;
          max_col = col;
        }
      }
      output[(row + output_offset_0) * output_stride_0 + (max_col + output_offset_1) * output_stride_1] = 1.0;
    }
  
    __kernel void elementwise_add_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;


      c[c_index] = a[a_index] + b[b_index];
    }
  
    __kernel void elementwise_add_scalar_r_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      double b,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      // print a memory length

      c[c_index] = a[a_index] + b;
    }
  
    __kernel void elementwise_add_scalar_l_mat_f64(
      double a,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      c[c_index] = a + b[b_index];
    }
  
    __kernel void elementwise_sub_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;


      c[c_index] = a[a_index] - b[b_index];
    }
  
    __kernel void elementwise_sub_scalar_r_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      double b,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      // print a memory length

      c[c_index] = a[a_index] - b;
    }
  
    __kernel void elementwise_sub_scalar_l_mat_f64(
      double a,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      c[c_index] = a - b[b_index];
    }
  
    __kernel void elementwise_mul_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;


      c[c_index] = a[a_index] * b[b_index];
    }
  
    __kernel void elementwise_mul_scalar_r_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      double b,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      // print a memory length

      c[c_index] = a[a_index] * b;
    }
  
    __kernel void elementwise_mul_scalar_l_mat_f64(
      double a,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      c[c_index] = a * b[b_index];
    }
  
    __kernel void elementwise_div_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;


      c[c_index] = a[a_index] / b[b_index];
    }
  
    __kernel void elementwise_div_scalar_r_mat_f64(
      __global const double* a,
      unsigned long a_stride_0, unsigned long a_stride_1,
      unsigned long a_offset_0, unsigned long a_offset_1,
      double b,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long a_index = (row + a_offset_0) * a_stride_0 + (col + a_offset_1) * a_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      // print a memory length

      c[c_index] = a[a_index] / b;
    }
  
    __kernel void elementwise_div_scalar_l_mat_f64(
      double a,
      __global const double* b,
      unsigned long b_stride_0, unsigned long b_stride_1,
      unsigned long b_offset_0, unsigned long b_offset_1,
      __global double* c,
      unsigned long c_stride_0, unsigned long c_stride_1,
      unsigned long c_offset_0, unsigned long c_offset_1,
      unsigned long rows, unsigned long cols
    ) {

      unsigned long row = get_global_id(0);
      unsigned long col = get_global_id(1);

      if (row >= rows || col >= cols) return;

      unsigned long b_index = (row + b_offset_0) * b_stride_0 + (col + b_offset_1) * b_stride_1;
      unsigned long c_index = (row + c_offset_0) * c_stride_0 + (col + c_offset_1) * c_stride_1;

      c[c_index] = a / b[b_index];
    }
  
// Kernels from tensor.rs (type: double)

    __kernel void im2col_f64(
      __global const double* input,
      unsigned long input_stride_0, unsigned long input_stride_1, unsigned long input_stride_2, unsigned long input_stride_3,
      unsigned long input_offset_0, unsigned long input_offset_1, unsigned long input_offset_2, unsigned long input_offset_3,
      __global double* output,
      unsigned long output_stride_0, unsigned long output_stride_1,
      unsigned long output_offset_0, unsigned long output_offset_1,
      unsigned long filter_stride_0, unsigned long filter_stride_1, unsigned long filter_stride_2,
      unsigned long channel_size, unsigned long col_size, unsigned long row_size,
      unsigned long stride_0, unsigned long stride_1, unsigned long stride_2
    ) {
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
    }
    
    #pragma OPENCL EXTENSION cl_khr_unsigned long64_base_atomics : enable
    #pragma OPENCL EXTENSION cl_khr_unsigned long64_extended_atomics : enable
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    void atomic_add_double(volatile global double *addr, double val) {
      union {
        double d;
        ulong ul;
      } old_val, new_val;
      old_val.ul = *((volatile global ulong*)addr);
      do {
        new_val.d = old_val.d + val;
        ulong read_val = atom_cmpxchg((volatile global ulong*)addr, old_val.ul, new_val.ul);
        if (read_val == old_val.ul) {
            break;
        }
        old_val.ul = read_val;
      } while (true);
    }
    __kernel void col2im_f64(
      __global const double* input,
      unsigned long input_stride_0, unsigned long input_stride_1,
      unsigned long input_offset_0, unsigned long input_offset_1,
      __global double* output,
      unsigned long output_stride_0, unsigned long output_stride_1, unsigned long output_stride_2, unsigned long output_stride_3,
      unsigned long output_offset_0, unsigned long output_offset_1, unsigned long output_offset_2, unsigned long output_offset_3,
      unsigned long filter_stride_0, unsigned long filter_stride_1, unsigned long filter_stride_2,
      unsigned long channel_size, unsigned long col_size, unsigned long row_size,
      unsigned long stride_0, unsigned long stride_1, unsigned long stride_2
    ) {
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
      atomic_add_double(&output[output_index], input[input_index]);
    }
    
    __kernel void padding_f64(
      __global const double* input,
      unsigned long N, unsigned long C, unsigned long H, unsigned long W,
      unsigned long input_stride_0, unsigned long input_stride_1, unsigned long input_stride_2, unsigned long input_stride_3,
      unsigned long input_offset_0, unsigned long input_offset_1, unsigned long input_offset_2, unsigned long input_offset_3,
      __global double* output,
      unsigned long C_out, unsigned long H_out, unsigned long W_out,
      unsigned long output_stride_0, unsigned long output_stride_1, unsigned long output_stride_2, unsigned long output_stride_3,
      unsigned long output_offset_0, unsigned long output_offset_1, unsigned long output_offset_2, unsigned long output_offset_3,
      int pad_front_c, int pad_back_c,
      int pad_top, int pad_bottom, int pad_left, int pad_right
    ) {
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

      double v = (double)0;

      if (c_in >= 0 && c_in < C
          && in_y >= 0 && in_y < H
          && in_x >= 0 && in_x < W) {
        unsigned long ii = input_offset_0 + n*input_stride_0
               + input_offset_1 + c_in*input_stride_1
               + input_offset_2 + in_y*input_stride_2
               + input_offset_3 + in_x*input_stride_3;
        v = input[ii];
      }

      unsigned long oo = output_offset_0 + n*output_stride_0
             + output_offset_1 + c_out*output_stride_1
             + output_offset_2 + y*output_stride_2
             + output_offset_3 + x*output_stride_3;
      output[oo] = v;
    }
    
    __kernel void relu_mask_tensor_f64(
      __global const double* input,
      __global const size_t* input_strides,
      __global double* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {0};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }
      unsigned long input_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {
        input_index += idx[i] * input_strides[i];
        output_index += idx[i] * output_strides[i];
      }

      output[output_index] = (input[input_index] > 0) ? (double)1 : (double)0;
    }
    
    __kernel void elementwise_add_tensor_f64(
      __global const double* input1,
      __global const size_t* input1_strides,
      __global const double* input2,
      __global const size_t* input2_strides,
      __global double* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {0};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }
      unsigned long input1_index = 0;
      unsigned long input2_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {
        input1_index += idx[i] * input1_strides[i];
        input2_index += idx[i] * input2_strides[i];
        output_index += idx[i] * output_strides[i];
      }

      output[output_index] = input1[input1_index] + input2[input2_index];
    }
    
    __kernel void elementwise_sub_tensor_f64(
      __global const double* input1,
      __global const size_t* input1_strides,
      __global const double* input2,
      __global const size_t* input2_strides,
      __global double* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {0};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }
      unsigned long input1_index = 0;
      unsigned long input2_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {
        input1_index += idx[i] * input1_strides[i];
        input2_index += idx[i] * input2_strides[i];
        output_index += idx[i] * output_strides[i];
      }

      output[output_index] = input1[input1_index] - input2[input2_index];
    }
    
    __kernel void elementwise_mul_tensor_f64(
      __global const double* input1,
      __global const size_t* input1_strides,
      __global const double* input2,
      __global const size_t* input2_strides,
      __global double* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {0};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }
      unsigned long input1_index = 0;
      unsigned long input2_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {
        input1_index += idx[i] * input1_strides[i];
        input2_index += idx[i] * input2_strides[i];
        output_index += idx[i] * output_strides[i];
      }

      output[output_index] = input1[input1_index] * input2[input2_index];
    }
    
    __kernel void elementwise_div_tensor_f64(
      __global const double* input1,
      __global const size_t* input1_strides,
      __global const double* input2,
      __global const size_t* input2_strides,
      __global double* output,
      __global const size_t* output_strides,
      unsigned long ndim
    ) {
      unsigned long grid = get_global_id(0);
      unsigned long idx[100] = {0};
      unsigned long rem = grid;
      for (unsigned long i = 0; i < ndim; i++) {
        idx[i] = rem / output_strides[i];
        rem %= output_strides[i];
      }
      unsigned long input1_index = 0;
      unsigned long input2_index = 0;
      unsigned long output_index = 0;
      for (unsigned long i = 0; i < ndim; i++) {
        input1_index += idx[i] * input1_strides[i];
        input2_index += idx[i] * input2_strides[i];
        output_index += idx[i] * output_strides[i];
      }

      output[output_index] = input1[input1_index] / input2[input2_index];
    }
    