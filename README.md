![tract-logo](assets/tract-logo/PNG/tract-horizontal-blue.png)

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![rustc >= 1.65.0](https://img.shields.io/badge/rustc-%3E%3D1.65.0-brightgreen)
![MIT/Apache 2](https://img.shields.io/crates/l/tract)
[![Native Linux test status](https://github.com/snipsco/tract/workflows/Native%20Linux/badge.svg)](https://github.com/snipsco/tract/actions)
[![Embedded targets status](https://github.com/snipsco/tract/workflows/Embedded%20targets/badge.svg)](https://github.com/snipsco/tract/actions)
[![Doc](https://docs.rs/tract-core/badge.svg)](https://docs.rs/tract-core)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://pypi.org/project/tract/)

## IC-AI

This fork of Tract focuses on leaning down the toolkit in order to run Neural Network Inference on the Internet Computer.
Components which are incompatible with the Internet Computer have been removed. They will be reintroduced with different designs when time allows. 
The goal of this fork is to increase the modularity and focus of Tract-Onnx.

## What ?

`tract` is a Neural Network inference toolkit. It can read ONNX or NNEF, optimize them and run them.

## Tract Documentation

There is also [some technical documentation](doc/) and [blog](https://tech-blog.sonos.com/posts/optimising-a-neural-network-for-inference/) posts.

## Tract in the landscape

### ONNX

As of today, `tract` passes successfully about 85% of ONNX backends
tests. All "real life" integration tests in ONNX test suite are passing: 
bvlc_alexnet, densenet121, inception_v1, inception_v2, resnet50, shufflenet,
squeezenet, vgg19, zfnet512.

Notable missing parts are operators dealing with Tensor Sequences and Optional Tensors : tract /really/ wants to flow Tensors and nothing else.
This is structural. Changing it would be pretty difficult, and it's unclear whether it can be done without impairing performance or maintainability.
We are not convinced these features have shown their interest in the wild yet, so we prefer to leave them aside.

Other dark corners are specific operators like "Resize" which fit perfectly in the framework but need a complex internal logic that is far
from our core business. In these cases, we are happy to accept contributions and to help. 

The following operators are implemented and tested.

Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, ArrayFeatureExtractor, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, BitShift, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BlackmanWindow, Cast, CastLike, CategoryMapper, Ceil, Clip, Compress, Concat, Constant, ConstantLike, ConstantOfShape, Conv, ConvInteger, ConvTranspose, Cos, Cosh, CumSum, DFT, DepthToSpace, DequantizeLinear, Div, Dropout, DynamicQuantizeLinear, Einsum, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, GRU, Gather, GatherElements, GatherND, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HammingWindow, HannWindow, HardSigmoid, Hardmax, Identity, If, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, MatMul, MatMulInteger, Max, MaxPool, Mean, MelWeightMatrix, Min, Mod, Mul, Multinomial, Neg, NonMaxSuppression, NonZero, Not, OneHot, Or, PRelu, Pad, ParametricSoftplus, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, RNN, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, Round, Rsqrt, STFT, ScaledTanh, Scan, Scatter, ScatterElements, ScatterND, Selu, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, Softplus, Softsign, SpaceToDepth, Split, Sqrt, Squeeze, Sub, Sum, Tan, Tanh, ThresholdedRelu, Tile, Transpose, TreeEnsembleClassifier, Unsqueeze, Where, Xor

We test these operators against from ONNX 1.4.1 (operator set 9), up to ONNX 1.13.0 (operator set 18).

We are using ONNX test suite, but it does not cover everything.
We also deliberately ignore some tests, or restricting their scope depending on what we feel is realistic.
Sometimes these decisions are just wrong, and sometimes they become wrong as time goes by and the fields moves in unexpected directions.
So if you are puzzled by an ONNX model that does not work in tract, we are happy to take a look.

### NNEF

Long story short, TensorFlow and ONNX formats are good for designing and
training networks. They need to move fast to follow the research field, tend to
integrate new features and operators greedily. They also exhibit a high level
of expressivity to facilitate network design.

On the other hand, only a subset of operators and network features actually
reach production, so systems running production network do not have to deal
with so many operators. Furthermore, some information required for training can
be stripped from the network before going to production for prediction.

NNEF tries to bridge the gap between training frameworks and inference by
proposing a format dedicated to production and prediction.

Tract supports NNEF:

* tract_nnef can load and execute NNEF networks
* tract supports most of the NNEF specification, the most notable exception
    being the ROI operators
* tract introduces tract-OPL, a series of NNEF extensions to support other
    operators (or extend some operators semantics) in order to represent the
    full range of tract-core neural network support: any network understood by
    tract should be serializable to tract-OPL. This is a work in progress.
* tract command line can translate networks from TensorFlow or ONNX to NNEF/OPL.

### tract-opl version compatibility

A remainder: NNEF is not expressive enough to represent all ONNX. tract-OPL extends
NNEF using proprietary to support what is missing. Notable extensions are pulse
operators, recurring operators (as Scan) and symbolic extensions.

There is no strict check in place here, so... implementation is not bullet proof.
* NNEF part aims at being very stable. It is strongly constrained with compatibility
with NNEF specification.
* tract-opl is a bit more in flux. Nevertheless we try to maintain the following
golden rule:

     `models serialized with tract 0.x.y should work with tract 0.x.z where z >= y`

* in practice, breaking changes have been relatively rare so far. Most models are
forward and retro compatible from when tract has acquired NNEF support.

Notable breakage occurred:
* 0.16.3 (forward compatible) on Scan operator
* 0.17.0 for binary decision tree classifier

Starting with `0.17.0`, a model property is injected in tract-opl files (`tract_nnef_ser_version`)
to tag which version of tract generated the file. As most models will remain compatible,
tract will not do any version check. It is up to the application developer to do so.

A softer version tag exists as `tract_nnef_format_version`. pre-0.17.0 version set it to
`alpha1`, post-0.17.0 set it `beta1`. Don't put too much emphasis into the "alpha-ness" naming 
of versions here.

# License

Note: files in the `onnx/protos` directory are copied from the
[ONNX](https://github.com/onnx/onnx) project and are not
covered by the following license statement.

## Apache 2.0/MIT

All original work licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

