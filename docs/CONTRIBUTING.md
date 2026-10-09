# 🤝 Contributing

We welcome contributions! Leann is built by the community, for the community.

## Ways to Contribute

- 🐛 **Bug Reports**: Found an issue? Let us know!
- 💡 **Feature Requests**: Have an idea? We'd love to hear it!
- 🔧 **Code Contributions**: PRs welcome for all skill levels
- 📖 **Documentation**: Help make Leann more accessible
- 🧪 **Benchmarks**: Share your performance results

## 🚀 Development Setup

### Prerequisites

1. **Install uv** (fast Python package installer):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/StarTrail-org/LEANN.git leann
   git submodule update --init --recursive
   cd leann
   ```

3. **Install system dependencies**:

   **macOS:**
   ```bash
   brew install llvm libomp boost protobuf zeromq pkgconf
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install libomp-dev libboost-all-dev protobuf-compiler \
                        libabsl-dev libmkl-full-dev libaio-dev libzmq3-dev
   ```

4. **Build from source**:
   ```bash
   # macOS
   CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ uv sync

   # Ubuntu/Debian
   uv sync
   ```

## 🔨 Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency. This runs automatically before each commit.

### Setup Pre-commit

1. **Install pre-commit tools**:
   ```bash
   uv sync --group lint
   ```

2. **Install the git hooks**:
   ```bash
   pre-commit install
   ```

3. **Run pre-commit manually** (optional):
   ```bash
   uv run pre-commit run --all-files
   ```

### Pre-commit Checks

Our pre-commit configuration includes:
- **Trailing whitespace removal**
- **End-of-file fixing**
- **YAML validation**
- **Large file prevention**
- **Merge conflict detection**
- **Debug statement detection**
- **Code formatting with ruff**
- **Code linting with ruff**

## 🧪 Testing

### Running Tests

```bash
# Install test tools only (no project runtime)
uv sync --group test

# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/test_filename.py

# Run with coverage
uv run pytest --cov=leann
```

### Writing Tests

- Place tests in the `test/` directory
- Follow the naming convention `test_*.py`
- Use descriptive test names that explain what's being tested
- Include both positive and negative test cases

## 📝 Code Style

We use `ruff` for both linting and formatting to ensure consistent code style.

### Format Your Code

```bash
# Format all files
ruff format

# Check formatting without changing files
ruff format --check
```

### Lint Your Code

```bash
# Run linter with auto-fix
ruff check --fix

# Just check without fixing
ruff check
```

### Style Guidelines

- Follow PEP 8 conventions
- Use descriptive variable names
- Add type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

## 🚦 CI/CD

Our CI pipeline runs automatically on all pull requests. It includes:

1. **Linting and Formatting**: Ensures code follows our style guidelines
2. **Multi-platform builds**: Tests on Ubuntu and macOS
3. **Python version matrix**: Tests on Python 3.10-3.14 (see the workflow for platform-specific coverage)
4. **Wheel building**: Ensures packages can be built and distributed

### CI Commands

The CI uses the same commands as pre-commit to ensure consistency:
```bash
# Linting
ruff check .

# Format checking
ruff format --check .
```

Make sure your code passes these checks locally before pushing!

## 🔄 Pull Request Process

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run pre-commit checks**:
   ```bash
   pre-commit run --all-files
   ```

4. **Test your changes**:
   ```bash
   uv run pytest
   ```

5. **Commit with descriptive messages**:
   ```bash
   git commit -m "feat: add new search algorithm"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `perf:` for performance improvements

6. **Push and create a pull request**:
   - Provide a clear description of your changes
   - Reference any related issues
   - Include examples or screenshots if applicable

## 📚 Documentation

When adding new features or making significant changes:

1. Update relevant documentation in `/docs`
2. Add docstrings to new functions/classes
3. Update README.md if needed
4. Include usage examples

## 🤔 Getting Help

- **Discord**: Join our community for discussions
- **Issues**: Check existing issues or create a new one
- **Discussions**: For general questions and ideas

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).

---

Thank you for contributing to LEANN! Every contribution, no matter how small, helps make the project better for everyone. 🌟
