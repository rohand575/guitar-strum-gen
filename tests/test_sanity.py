"""
Test Suite - Basic Sanity Tests

These tests verify that the package structure is correct
and that basic imports work.

Run with: pytest tests/test_sanity.py -v
"""

import pytest


class TestPackageStructure:
    """Test that all packages can be imported."""
    
    def test_import_src(self):
        """Test that main src package can be imported."""
        import src
        assert hasattr(src, "__version__")
        assert src.__version__ == "0.1.0"
    
    def test_import_data_package(self):
        """Test that data subpackage can be imported."""
        import src.data
        # Will have more specific tests when schema.py is created
    
    def test_import_rules_package(self):
        """Test that rules subpackage can be imported."""
        import src.rules
    
    def test_import_models_package(self):
        """Test that models subpackage can be imported."""
        import src.models
    
    def test_import_train_package(self):
        """Test that train subpackage can be imported."""
        import src.train
    
    def test_import_evaluation_package(self):
        """Test that evaluation subpackage can be imported."""
        import src.evaluation
    
    def test_import_app_package(self):
        """Test that app subpackage can be imported."""
        import src.app


class TestProjectPaths:
    """Test that project paths are set up correctly."""
    
    def test_data_directories_exist(self):
        """Test that data directories can be referenced."""
        from pathlib import Path
        # These tests assume we're running from project root
        # In practice, we'll use absolute paths or config
        pass  # Placeholder - will be expanded
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
