// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics
func.func @broadcast(%ctx: !hip.context, %y: tensor<2x1xf16>, %x: tensor<1x3xf16>, %out: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %r = hip.atan2(%ctx) ins(%y, %x : tensor<2x1xf16>, tensor<1x3xf16>) outs(%out : tensor<2x3xf16>) : tensor<2x3xf16>
  return %r : tensor<2x3xf16>
}
// -----
func.func @integer(%ctx: !hip.context, %y: tensor<2xi32>, %x: tensor<2xi32>, %out: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{requires f16 or f32 elements}}
  %r = hip.atan2(%ctx) ins(%y, %x : tensor<2xi32>, tensor<2xi32>) outs(%out : tensor<2xi32>) : tensor<2xi32>
  return %r : tensor<2xi32>
}
// -----
func.func @invalid_broadcast(%ctx: !hip.context, %y: tensor<2xf32>, %x: tensor<3xf32>, %out: tensor<3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{input shape cannot broadcast to output}}
  %r = hip.atan2(%ctx) ins(%y, %x : tensor<2xf32>, tensor<3xf32>) outs(%out : tensor<3xf32>) : tensor<3xf32>
  return %r : tensor<3xf32>
}
// -----
func.func @rank5(%ctx: !hip.context, %y: tensor<1x1x1x1x2xf32>, %x: tensor<1x1x1x1x2xf32>, %out: tensor<1x1x1x1x2xf32>) -> tensor<1x1x1x1x2xf32> {
  // expected-error @+1 {{requires ranked tensors/memrefs of rank at most 4}}
  %r = hip.atan2(%ctx) ins(%y, %x : tensor<1x1x1x1x2xf32>, tensor<1x1x1x1x2xf32>) outs(%out : tensor<1x1x1x1x2xf32>) : tensor<1x1x1x1x2xf32>
  return %r : tensor<1x1x1x1x2xf32>
}
