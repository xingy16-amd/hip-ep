/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include <cstdio>

int wrap_atan2(RuntimeState *state, void *lhs, void *rhs, void *output,
               int64_t lhs_n, int64_t lhs_c, int64_t lhs_h, int64_t lhs_w,
               int64_t rhs_n, int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
               int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
               int64_t data_type) {
  OP_PROFILE(
      "atan2",
      [&] {
        return std::to_string(out_n) + "x" + std::to_string(out_c) + "x" +
               std::to_string(out_h) + "x" + std::to_string(out_w);
      },
      state);
  if (!state || !lhs || !rhs || !output)
    return -1;
  int dtype;
  if (data_type == HIPDNN_EP_DATATYPE_HALF)
    dtype = HIP_DTYPE_FLOAT16;
  else if (data_type == HIPDNN_EP_DATATYPE_FLOAT)
    dtype = HIP_DTYPE_FLOAT32;
  else
    return -1;
  const int64_t lhs_shape[] = {lhs_n, lhs_c, lhs_h, lhs_w};
  const int64_t rhs_shape[] = {rhs_n, rhs_c, rhs_h, rhs_w};
  const int64_t out_shape[] = {out_n, out_c, out_h, out_w};
  int rc = hip_elementwise_binary_bcast(hipdnn_ep_state_get_stream(state), lhs,
                                        rhs, output, lhs_shape, rhs_shape,
                                        out_shape, 4, dtype);
  if (rc != 0)
    fprintf(stderr, "wrap_atan2: broadcast kernel failed (%d)\n", rc);
  return rc;
}
