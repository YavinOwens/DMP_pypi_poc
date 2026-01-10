"""
Tests for package imports
"""



class TestImports:
    """Test that all main imports work correctly"""

    def test_main_package_import(self):
        """Test that main package can be imported"""
        import datamanagement_genai
        assert hasattr(datamanagement_genai, '__version__')
        assert datamanagement_genai.__version__ == "0.1.4"

    def test_core_imports(self):
        """Test that core functions can be imported"""
        from datamanagement_genai import (
            create_word_document,
            generate_llm_report,
            get_snowflake_session,
            run_model_benchmarks,
        )
        assert callable(get_snowflake_session)
        assert callable(run_model_benchmarks)
        assert callable(generate_llm_report)
        assert callable(create_word_document)

    def test_rag_system_import(self):
        """Test that RAGSystem can be imported"""
        from datamanagement_genai import RAGSystem
        assert RAGSystem is not None

    def test_data_quality_import(self):
        """Test that DataQualityRulesManager can be imported"""
        from datamanagement_genai import DataQualityRulesManager
        assert DataQualityRulesManager is not None
