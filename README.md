# Fogo Memory Pair - Standalone Package

A standalone memory pair software package for machine unlearning research. This package provides the core `StreamNewtonMemoryPair` implementation that can be easily integrated into other repositories for testing and comparison with other unlearning methods.

## Features

- **StreamNewtonMemoryPair**: Main memory pair implementation for machine unlearning
- **LimitedMemoryBFGS**: Online L-BFGS optimization backend  
- **Privacy-preserving deletion**: (ε,δ)-differential privacy guarantees
- **Event-based logging**: Comprehensive logging infrastructure for experiments
- **Standalone usage**: Can be cloned/copied into any repository

## Quick Start

### As a standalone package

1. **Clone or copy the `src/` directory** into your repository
2. **Install dependencies**:
   ```bash
   pip install numpy scipy structlog python-json-logger PyYAML
   ```
3. **Import and use**:
   ```python
   from src.memory_pair import StreamNewtonMemoryPair
   
   # Initialize memory pair
   memory_pair = StreamNewtonMemoryPair(
       dim=10,                # feature dimension
       lam=0.1,              # ridge regularization
       eps_total=1.0,        # privacy budget
       delta_total=1e-5,     # privacy parameter
       max_deletions=20      # max number of deletions
   )
   
   # Insert data points
   memory_pair.insert(x, y)
   
   # Unlearn data points
   memory_pair.delete(x, y)
   ```

### As an installed package

1. **Install the package**:
   ```bash
   pip install -e .
   ```
2. **Import and use**:
   ```python
   from src import StreamNewtonMemoryPair
   
   # Same usage as above
   ```

## Example Usage

See `example_usage.py` for a complete example showing:
- Basic machine unlearning workflow
- Comparison with retraining from scratch
- Framework for comparing different unlearning methods

Run the example:
```bash
python example_usage.py
```

## API Reference

### StreamNewtonMemoryPair

The main class for memory pair-based machine unlearning.

#### Constructor
```python
StreamNewtonMemoryPair(
    dim: int,
    lam: float = 1.0,
    eps_total: float = 1.0,
    delta_total: float = 1e-5,
    max_deletions: int = 20
)
```

**Parameters:**
- `dim`: Feature dimension
- `lam`: Ridge regularization parameter
- `eps_total`: Total privacy budget for all deletions
- `delta_total`: Total δ budget for all deletions  
- `max_deletions`: Maximum number of delete() calls

#### Methods

**`insert(x: np.ndarray, y: float)`**
- Insert a new data point and perform one Newton step
- Updates model parameters using Sherman-Morrison formula

**`delete(x: np.ndarray, y: float)`**
- Remove influence of a data point (machine unlearning)
- Adds calibrated Gaussian noise for privacy
- Requires the exact (x, y) pair to be provided

**`privacy_ok() -> bool`**
- Returns True if privacy budget is not exceeded

#### Properties
- `theta`: Current model parameters
- `eps_spent`: Privacy budget used so far
- `deletions_so_far`: Number of deletions performed

### LimitedMemoryBFGS

L-BFGS optimization backend with curvature pair management.

#### Constructor
```python
LimitedMemoryBFGS(m_max: int = 10)
```

#### Methods
- `add_pair(s, y)`: Add curvature pair
- `direction(g)`: Compute search direction
- `remove_pair_at(idx)`: Remove curvature pair

## File Structure

```
src/
├── __init__.py              # Package initialization
├── memory_pair.py           # Main StreamNewtonMemoryPair class
├── l_bfgs.py               # L-BFGS optimization implementation
└── event_logging/          # Logging infrastructure
    ├── __init__.py
    └── config.yaml
```

## Integration with Other Repositories

This package is designed to be easily integrated into machine unlearning research:

1. **Copy the `src/` directory** into your project
2. **Install dependencies** as listed above
3. **Use in your experiments**:
   ```python
   from src.memory_pair import StreamNewtonMemoryPair
   
   # Your experiment code here
   # Compare with other unlearning methods
   ```

## Testing

Run the test suite to verify functionality:
```bash
python test_standalone.py
```

This tests:
- Basic memory pair functionality
- L-BFGS optimization
- Complete unlearning workflow
- Privacy budget management

## Dependencies

- `numpy>=1.21`: Numerical computing
- `scipy>=1.7`: Scientific computing
- `structlog>=21.0`: Structured logging
- `python-json-logger>=2.0`: JSON log formatting
- `PyYAML>=6.0`: YAML configuration

## License

MIT License - see the project root for details.

## Citation

If you use this package in your research, please cite:
```
@software{fogo_memory_pair,
  title={Fogo Memory Pair: Online Machine Unlearning},
  author={Kennon Stewart},
  year={2024},
  url={https://github.com/kennonstewart/Fogo}
}
```