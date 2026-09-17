// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
//
// The trailing i32 on the wrap_group_query_attention_right_padded_prefill call is op_state_slot,
// threaded by --assign-op-state-slots. It selects this GQA instance's
// per-instance GqaState (hipBLASLt GEMM descriptor cache), replacing the
// former shared RuntimeState::gqa_gemm_cache.

module {
  func.func @test_gqa_right_padding(%ctx: !hip.context,
                                %query: memref<1x8x4096xf16, 1>,
                                %key: memref<1x8x1024xf16, 1>,
                                %value: memref<1x8x1024xf16, 1>,
                                %past_key: memref<1x8x127x128xf16, 1>,
                                %past_value: memref<1x8x127x128xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x8x4096xf16, 1>,
                                %present_key: memref<1x8x128x128xf16, 1>,
                                %present_value: memref<1x8x128x128xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %seqlens_k, %total_seq_len :
            memref<1x8x4096xf16, 1>, memref<1x1xi32, 1>, memref<i32, 1>)
        key(%key : memref<1x8x1024xf16, 1>)
        value(%value : memref<1x8x1024xf16, 1>)
        past_key(%past_key : memref<1x8x127x128xf16, 1>)
        past_value(%past_value : memref<1x8x127x128xf16, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x8x4096xf16, 1>, memref<1x8x128x128xf16, 1>, memref<1x8x128x128xf16, 1>)
        {num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64, right_padded_prefill = true}
    return
  }
}

// CHECK-LABEL: llvm.func @test_gqa_right_padding
// CHECK: llvm.call @wrap_group_query_attention_right_padded_prefill({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, f32, i64, i64, i64, i64, i64, i64, i32, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// CHECK: llvm.call @hipdnn_ep_state_record_error
