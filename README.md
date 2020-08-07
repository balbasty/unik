# Unik

*A unified seamless interface for Keras, Tensorflow and Numpy.*

## Purpose

Developing models with Keras usually implies navigating between functions
that work on NumPy arrays, functions that that work on Tensorflow tensors
(symbolic or not) and layers that work on Keras tensors (always symbolic,
usually with a batch dimension).

This dichotomy often forces the same algorithms to be implemented multiple
times, within different framework, whether they are intended to work on values
that are only known symbolically (computational graph) or statically
(metaprogramming of the computational graph).

**Unik**  offers a collection of function that take and return any of these
tensor types (numpy, tensorflow, keras) and dispataches to the appropriate
implementation accordingly. These functions can in turn be used as building
blocks for type-agnostic algorithms, greatly easing model writing.

## Disclaimer

**Unik** is in a very alpha stage of development. Questions and bug reports
are welcome.

## License

MIT
