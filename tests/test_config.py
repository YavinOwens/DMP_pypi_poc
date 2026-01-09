"""
Tests for configuration management
"""

from datamanagement_genai.config import get_models, get_test_prompts, load_test_config


class TestConfig:
    """Test configuration loading and access"""

    def test_load_test_config_exists(self):
        """Test that test_config.json can be loaded"""
        config = load_test_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_get_models(self):
        """Test that models can be retrieved from config"""
        models = get_models()
        assert models is not None
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_get_test_prompts(self):
        """Test that test prompts can be retrieved from config"""
        prompts = get_test_prompts()
        assert prompts is not None
        assert isinstance(prompts, dict)
