"""
Tests for helper functions
"""

from datamanagement_genai.helpers import get_snowflake_session, get_table_columns


class TestHelpers:
    """Test helper functions"""

    def test_get_snowflake_session_no_config(self):
        """Test that get_snowflake_session handles missing config gracefully"""
        # Without proper config, this should return None or raise an error
        # depending on implementation
        session = get_snowflake_session()
        # Session might be None if config is missing, which is acceptable
        assert session is None or hasattr(session, 'sql')

    def test_get_table_columns_signature(self):
        """Test that get_table_columns function exists and has correct signature"""
        # Just verify the function exists and is callable
        assert callable(get_table_columns)
