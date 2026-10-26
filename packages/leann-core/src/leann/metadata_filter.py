"""
Metadata filtering engine for LEANN search results.

This module provides generic metadata filtering capabilities that can be applied
to search results from any LEANN backend. The filtering supports various
operators for different data types including numbers, strings, booleans, and lists.
"""

import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Type alias for filter specifications
FilterValue = Union[str, int, float, bool, list]
FilterSpec = dict[str, FilterValue]
MetadataFilters = dict[str, FilterSpec]


class MetadataFilterEngine:
    """
    Engine for evaluating metadata filters against search results.

    Supports various operators for filtering based on metadata fields:
    - Comparison: ==, !=, <, <=, >, >=
    - Membership: in, not_in
    - String operations: contains, starts_with, ends_with
    - Boolean operations: is_true, is_false
    """

    def __init__(self):
        """Initialize the filter engine with supported operators."""
        self.operators = {
            "==": self._equals,
            "!=": self._not_equals,
            "<": self._less_than,
            "<=": self._less_than_or_equal,
            ">": self._greater_than,
            ">=": self._greater_than_or_equal,
            "in": self._in,
            "not_in": self._not_in,
            "contains": self._contains,
            "starts_with": self._starts_with,
            "ends_with": self._ends_with,
            "is_true": self._is_true,
            "is_false": self._is_false,
        }

    def apply_filters(
        self, search_results: list[dict[str, Any]], metadata_filters: Optional[MetadataFilters]
    ) -> list[dict[str, Any]]:
        """
        Apply metadata filters to a list of search results.

        Args:
            search_results: List of result dictionaries, each containing 'metadata' field
            metadata_filters: Dictionary of filter specifications
                Format: {"field_name": {"operator": value}}

        Returns:
            Filtered list of search results
        """
        if not metadata_filters:
            return search_results

        logger.debug(f"Applying filters: {metadata_filters}")
        logger.debug(f"Input results count: {len(search_results)}")

        filtered_results = []
        for result in search_results:
            if self._evaluate_filters(result, metadata_filters):
                filtered_results.append(result)

        logger.debug(f"Filtered results count: {len(filtered_results)}")
        return filtered_results

    def _evaluate_filters(self, result: dict[str, Any], filters: MetadataFilters) -> bool:
        """
        Evaluate all filters against a single search result.

        All filters must pass (AND logic) for the result to be included.

        Args:
            result: Full search result dictionary (including metadata, text, etc.)
            filters: Filter specifications to evaluate

        Returns:
            True if all filters pass, False otherwise
        """
        for field_name, filter_spec in filters.items():
            if not self._evaluate_field_filter(result, field_name, filter_spec):
                return False
        return True

    def _evaluate_field_filter(
        self, result: dict[str, Any], field_name: str, filter_spec: FilterSpec
    ) -> bool:
        """
        Evaluate a single field filter against a search result.

        Args:
            result: Full search result dictionary
            field_name: Name of the field to filter on
            filter_spec: Filter specification for this field

        Returns:
            True if the filter passes, False otherwise
        """
        # First check top-level fields, then check metadata
        field_value = result.get(field_name)
        if field_value is None:
            # Try to get from metadata if not found at top level
            metadata = result.get("metadata", {})
            field_value = metadata.get(field_name)

        # Handle missing fields - they fail all filters except existence checks
        if field_value is None:
            logger.debug(f"Field '{field_name}' not found in result or metadata")
            return False

        # Evaluate each operator in the filter spec
        for operator, expected_value in filter_spec.items():
            if operator not in self.operators:
                logger.warning(f"Unsupported operator: {operator}")
                return False

            try:
                if not self.operators[operator](field_value, expected_value):
                    logger.debug(
                        f"Filter failed: {field_name} {operator} {expected_value} "
                        f"(actual: {field_value})"
                    )
                    return False
            except Exception as e:
                logger.warning(
                    f"Error evaluating filter {field_name} {operator} {expected_value}: {e}"
                )
                return False

        return True

    # Comparison operators
    def _equals(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value equals expected value."""
        return field_value == expected_value

    def _not_equals(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value does not equal expected value."""
        return field_value != expected_value

    def _less_than(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is less than expected value."""
        return self._numeric_compare(field_value, expected_value, lambda a, b: a < b)

    def _less_than_or_equal(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is less than or equal to expected value."""
        return self._numeric_compare(field_value, expected_value, lambda a, b: a <= b)

    def _greater_than(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is greater than expected value."""
        return self._numeric_compare(field_value, expected_value, lambda a, b: a > b)

    def _greater_than_or_equal(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is greater than or equal to expected value."""
        return self._numeric_compare(field_value, expected_value, lambda a, b: a >= b)

    # Membership operators
    def _in(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is in the expected list/collection."""
        if not isinstance(expected_value, (list, tuple, set)):
            raise ValueError("'in' operator requires a list, tuple, or set")
        if isinstance(field_value, (list, tuple, set)):
            return any(item in expected_value for item in field_value)
        return field_value in expected_value

    def _not_in(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is not in the expected list/collection."""
        if not isinstance(expected_value, (list, tuple, set)):
            raise ValueError("'not_in' operator requires a list, tuple, or set")
        return not self._in(field_value, expected_value)

    # String operators
    def _contains(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value contains the expected substring."""
        field_str = str(field_value)
        expected_str = str(expected_value)
        return expected_str in field_str

    def _starts_with(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value starts with the expected prefix."""
        field_str = str(field_value)
        expected_str = str(expected_value)
        return field_str.startswith(expected_str)

    def _ends_with(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value ends with the expected suffix."""
        field_str = str(field_value)
        expected_str = str(expected_value)
        return field_str.endswith(expected_str)

    # Boolean operators
    def _is_true(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is truthy."""
        return bool(field_value)

    def _is_false(self, field_value: Any, expected_value: Any) -> bool:
        """Check if field value is falsy."""
        return not bool(field_value)

    # Helper methods
    def _numeric_compare(self, field_value: Any, expected_value: Any, compare_func) -> bool:
        """
        Helper for numeric comparisons with type coercion.

        Args:
            field_value: Value from metadata
            expected_value: Value to compare against
            compare_func: Comparison function to apply

        Returns:
            Result of comparison
        """
        try:
            # Try to convert both values to numbers for comparison
            if isinstance(field_value, str) and isinstance(expected_value, str):
                # Numeric metadata can also be stored as strings. Keep lexical
                # ordering for dates and other strings that are not numbers.
                try:
                    field_decimal = Decimal(field_value)
                    expected_decimal = Decimal(expected_value)
                    if field_decimal.is_finite() and expected_decimal.is_finite():
                        return compare_func(field_decimal, expected_decimal)
                except InvalidOperation:
                    pass
                return compare_func(field_value, expected_value)

            # Numeric comparison - attempt to convert to float
            field_num = (
                float(field_value) if not isinstance(field_value, (int, float)) else field_value
            )
            expected_num = (
                float(expected_value)
                if not isinstance(expected_value, (int, float))
                else expected_value
            )

            return compare_func(field_num, expected_num)
        except (ValueError, TypeError):
            # Fall back to string comparison if numeric conversion fails
            return compare_func(str(field_value), str(expected_value))
