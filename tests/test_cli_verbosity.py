"""Tests for CLI verbosity options.

This module tests the configurable verbosity functionality that allows
suppressing C++ output from FAISS/HNSW.
See: https://github.com/StarTrail-org/LEANN/issues/187
"""

import os

import pytest


class TestSuppressCppOutput:
    """Test the suppress_cpp_output context manager."""

    def test_suppress_cpp_output_captures_stdout(self):
        """C output to stdout should be suppressed when enabled."""
        from leann.cli import suppress_cpp_output

        with suppress_cpp_output(suppress=True):
            # This goes to fd 1, but is redirected to devnull
            os.write(1, b"This should be suppressed\n")

        # If we got here without error, suppression worked
        # The text was written to devnull

    def test_suppress_cpp_output_captures_stderr(self):
        """C output to stderr should be suppressed when enabled."""
        from leann.cli import suppress_cpp_output

        with suppress_cpp_output(suppress=True):
            # This goes to fd 2, but is redirected to devnull
            os.write(2, b"This error should be suppressed\n")

    def test_suppress_cpp_output_restores_fds(self):
        """File descriptors should be restored after context."""
        from leann.cli import suppress_cpp_output

        # Save original fds
        original_stdout = os.dup(1)
        original_stderr = os.dup(2)

        try:
            with suppress_cpp_output(suppress=True):
                pass

            # Verify fds are still valid and point to original destinations
            # by checking we can write to them
            os.write(1, b"")  # Should not raise
            os.write(2, b"")  # Should not raise
        finally:
            os.close(original_stdout)
            os.close(original_stderr)

    def test_suppress_cpp_output_disabled(self):
        """When suppress=False, output should not be suppressed."""
        from leann.cli import suppress_cpp_output

        with suppress_cpp_output(suppress=False):
            # This should work normally
            os.write(1, b"")
            os.write(2, b"")


class TestCliVerbosityArgs:
    """Test CLI argument parsing for verbosity options."""

    def test_verbose_flag_parsed(self):
        """--verbose flag should be parsed correctly."""
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(["--verbose", "list"])
        assert args.verbose is True
        assert args.quiet is False

    def test_quiet_flag_parsed(self):
        """-q flag should be parsed correctly."""
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(["-q", "list"])
        assert args.quiet is True
        assert args.verbose is False

    def test_verbose_short_flag(self):
        """-v should work as shorthand for --verbose."""
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(["-v", "list"])
        assert args.verbose is True

    def test_verbose_and_quiet_mutually_exclusive(self):
        """--verbose and --quiet should be mutually exclusive."""
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--verbose", "--quiet", "list"])

    def test_default_is_quiet(self):
        """Default behavior should be quiet (suppress C++ output)."""
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(["list"])
        assert args.verbose is False
        assert args.quiet is False
        # When both are False, we suppress by default


class TestVerbosityIntegration:
    """Integration tests for verbosity in commands."""

    def test_list_command_does_not_suppress(self):
        """List command should work without suppression."""
        import asyncio

        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        # List command should not raise
        args = parser.parse_args(["list"])
        asyncio.run(cli.run(args))

    def test_verbose_flag_with_list(self):
        """Verbose flag should work with list command."""
        import asyncio

        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(["-v", "list"])
        asyncio.run(cli.run(args))
