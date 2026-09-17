// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s
// Main's captured tensors stay live across a loop call. Helper allocations
// must use a different runtime pool even when both have local domain zero.
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.get_pool({{.*}}) : memref<?xi8>
// CHECK-LABEL: func.func private @main_loop_body_0
// CHECK: hip.get_pool({{.*}}) {domain_id = 1 : i64} : memref<?xi8>
// CHECK-LABEL: func.func private @main_loop_body_1
// CHECK: hip.get_pool({{.*}}) {domain_id = 2 : i64} : memref<?xi8>
module {
  func.func @main_graph(%ctx: !hip.context, %x: memref<8xf32>) -> memref<8xf32> {
    %y = memref.alloc() : memref<8xf32>
    memref.copy %x, %y : memref<8xf32> to memref<8xf32>
    return %y : memref<8xf32>
  }
  func.func private @main_loop_body_0(%ctx: !hip.context, %x: memref<8xf32>) -> memref<8xf32> {
    %y = memref.alloc() : memref<8xf32>
    memref.copy %x, %y : memref<8xf32> to memref<8xf32>
    return %y : memref<8xf32>
  }
  func.func private @main_loop_body_1(%ctx: !hip.context, %x: memref<8xf32>) -> memref<8xf32> {
    %y = memref.alloc() : memref<8xf32>
    memref.copy %x, %y : memref<8xf32> to memref<8xf32>
    return %y : memref<8xf32>
  }
}
