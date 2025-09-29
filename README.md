<div align="center">
  <picture>
    <img alt="Orys Logo" src="./assets/orys.png" width="280"/>
  </picture>
</div>

<h3 align="center">
  Lightweight Edge AI Inference Runtime
</h3>
<br/>
<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  </a>
  <img src="https://img.shields.io/badge/status-alpha-red" alt="Alpha">
  <img src="https://img.shields.io/badge/rust-1.70+-orange" alt="Rust 1.70+">
</p>

<div align="center">
  <span>
    High-performance, memory-safe inference runtime for neural networks. Built in Rust for edge deployment with support for JSON models and Python bindings.
  </span>
</div>

---

## How It Works

Orys enables efficient neural network inference through a modular runtime:
- **Load models** from JSON format with automatic validation
- **Execute graphs** using topological sorting for optimal performance  
- **Handle tensors** with NumPy-style broadcasting operations
- **Deploy anywhere** with zero-dependency static binaries

## Key Features

### **Core Operators**
- MatMul, Add (with broadcasting), ReLU, Sigmoid
- Memory-safe tensor operations with shape validation
- Optimized for CPU inference on edge devices

### **Model Loading**
- JSON format with human-readable schema
- Automatic format detection and validation
- Extensible loader system for future formats

### **Runtime Engine** 
- Computation graph with topological execution
- Named tensor I/O with comprehensive error handling
- Execution statistics and debugging support

### **Python Integration**
- NumPy array compatibility with automatic conversion
- Pythonic API for model loading and inference
- Built with PyO3 for zero-copy data transfer

### **Edge Optimized**
- Minimal runtime dependencies
- Static linking for portable deployment
- Configurable feature flags for size optimization

## Quick Start

### Rust
```rust
use orys::prelude::*;

let mut inputs = HashMap::new();
inputs.insert("input".to_string(), ones(vec![1, 784]));

let outputs = run_model("model.json", inputs)?;
let prediction = &outputs["output"];
```

### Python
```python
import orys
import numpy as np

inputs = {"input": np.ones((1, 784), dtype=np.float32)}
outputs = orys.run_model("model.json", inputs)
print(outputs["output"])
```

## Development Status

> [!WARNING]
> Orys is in **ALPHA** and should be considered experimental.
> API may change between versions.

## Installation

### Rust
```bash
cargo add orys
```

### Python
```bash
pip install maturin
cd python-bindings
maturin develop
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## Built With

- **Language**: [Rust](https://rust-lang.org/) - Memory safety and performance
- **Bindings**: [PyO3](https://pyo3.rs/) - Python integration
- **Serialization**: [Serde](https://serde.rs/) - Fast JSON parsing
- **Tensors**: Custom implementation with broadcasting support

## Star History

Thanks for the crazy support 💖

[![Star History Chart](https://api.star-history.com/svg?repos=amosehiguese/orys&type=Date)](https://star-history.com/#amosehiguese/orys&Date)