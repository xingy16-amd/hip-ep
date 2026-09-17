// RUN: hip-mlir-opt --split-input-file --hip-pool-allocs='alignment=1' %s | FileCheck %s
// HIP boolean kernels write one byte per element. Packing 4097 booleans
// into ceil(4097/8) bytes corrupts adjacent buffers in large masks.
// CHECK: hipdnn.pool_size = 4097 : i64
// CHECK-LABEL: func.func @static_bool
// CHECK: arith.constant 4097 : index
// CHECK: hip.get_pool
func.func @static_bool(%ctx: !hip.context, %x: memref<4097xi1>) -> memref<4097xi1> {
  %out = memref.alloc() : memref<4097xi1>
  memref.copy %x, %out : memref<4097xi1> to memref<4097xi1>
  return %out : memref<4097xi1>
}

// -----

// The dynamic coefficient must also be bytes per element, not packed bits.
// CHECK-LABEL: func.func @dynamic_bool
// CHECK: arith.constant 3 : index
// CHECK: arith.muli
// CHECK: hip.get_pool
func.func @dynamic_bool(%ctx: !hip.context, %x: memref<?x3xi1>, %n: index) -> memref<?x3xi1> {
  %out = memref.alloc(%n) : memref<?x3xi1>
  memref.copy %x, %out : memref<?x3xi1> to memref<?x3xi1>
  return %out : memref<?x3xi1>
}
