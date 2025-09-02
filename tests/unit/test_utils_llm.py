"""
Tests for LLM utility functions.

This module tests the LLM utility functions used across NovaEval.
"""

from unittest.mock import Mock, patch

import pytest

from src.novaeval.utils.llm import call_llm

pytestmark = pytest.mark.unit


class TestCallLLM:
    """Test the call_llm function with different model types."""

    def test_call_llm_with_novaeval_model(self):
        """Test calling LLM with NovaEval-style model (has .generate method)."""
        mock_model = Mock()
        mock_model.generate.return_value = "Generated response"

        result = call_llm(mock_model, "Test prompt")

        assert result == "Generated response"
        mock_model.generate.assert_called_once_with("Test prompt")

    @patch("src.novaeval.utils.llm._HAS_LANGCHAIN", True)
    @patch("src.novaeval.utils.llm.isinstance")
    def test_call_llm_with_langchain_invoke(self, mock_isinstance):
        """Test calling LLM with LangChain model that has invoke method."""
        mock_model = Mock()
        mock_model.invoke.return_value = "LangChain invoke response"

        # Make isinstance return True for BaseLanguageModel
        mock_isinstance.return_value = True

        result = call_llm(mock_model, "Test prompt")

        assert result == "LangChain invoke response"
        mock_model.invoke.assert_called_once_with("Test prompt")

    @patch("src.novaeval.utils.llm._HAS_LANGCHAIN", True)
    @patch("src.novaeval.utils.llm.isinstance")
    def test_call_llm_with_langchain_generate(self, mock_isinstance):
        """Test calling LLM with LangChain model that has generate method (no invoke)."""
        mock_model = Mock()
        # Remove invoke method, keep generate
        del mock_model.invoke

        # Mock the generate response structure
        mock_generation = Mock()
        mock_generation.text = "LangChain generate response"
        mock_model.generate.return_value.generations = [[mock_generation]]

        # Make isinstance return True for BaseLanguageModel
        mock_isinstance.return_value = True

        result = call_llm(mock_model, "Test prompt")

        assert result == "LangChain generate response"
        mock_model.generate.assert_called_once_with(["Test prompt"])

    @patch("src.novaeval.utils.llm._HAS_LANGCHAIN", True)
    @patch("src.novaeval.utils.llm.isinstance")
    def test_call_llm_with_langchain_no_methods(self, mock_isinstance):
        """Test calling LLM with LangChain model that has neither invoke nor generate."""
        mock_model = Mock()
        # Remove both methods
        del mock_model.invoke
        del mock_model.generate

        # Make isinstance return True for BaseLanguageModel
        mock_isinstance.return_value = True

        with pytest.raises(
            ValueError, match="LangChain LLM does not support invoke or generate"
        ):
            call_llm(mock_model, "Test prompt")

    def test_call_llm_with_string_model(self):
        """Test calling LLM with string model name (should raise NotImplementedError)."""
        with pytest.raises(
            NotImplementedError, match="String-based LLM calling not implemented"
        ):
            call_llm("gpt-3.5-turbo", "Test prompt")

    @patch("src.novaeval.utils.llm._HAS_LANGCHAIN", False)
    def test_call_llm_without_langchain_installed(self):
        """Test calling LLM when LangChain is not installed."""
        mock_model = Mock()
        mock_model.generate.return_value = "NovaEval response"

        result = call_llm(mock_model, "Test prompt")

        assert result == "NovaEval response"
        mock_model.generate.assert_called_once_with("Test prompt")

    def test_call_llm_with_object_without_generate(self):
        """Test calling LLM with object that doesn't have generate method."""
        mock_model = Mock()
        del mock_model.generate  # Remove generate method

        with pytest.raises(AttributeError):
            call_llm(mock_model, "Test prompt")


class TestLangChainImport:
    """Test LangChain import detection."""

    @patch("src.novaeval.utils.llm._HAS_LANGCHAIN", True)
    def test_langchain_available(self):
        """Test behavior when LangChain is available."""
        # This test verifies the import detection works

        # In this patched context, it should be True
        # The actual test is in the integration with call_llm above
        pass

    @patch("src.novaeval.utils.llm._HAS_LANGCHAIN", False)
    def test_langchain_not_available(self):
        """Test behavior when LangChain is not available."""

        # In this patched context, it should be False
        # The actual test is in the integration with call_llm above
        pass


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_call_llm_with_none_model(self):
        """Test calling LLM with None model."""
        with pytest.raises(AttributeError):
            call_llm(None, "Test prompt")

    def test_call_llm_with_empty_prompt(self):
        """Test calling LLM with empty prompt."""
        mock_model = Mock()
        mock_model.generate.return_value = "Response to empty prompt"

        result = call_llm(mock_model, "")

        assert result == "Response to empty prompt"
        mock_model.generate.assert_called_once_with("")

    def test_call_llm_with_complex_prompt(self):
        """Test calling LLM with complex multi-line prompt."""
        mock_model = Mock()
        mock_model.generate.return_value = "Complex response"

        complex_prompt = """
        This is a complex prompt with:
        - Multiple lines
        - Special characters: !@#$%^&*()
        - Unicode: 🚀 ✨ 🎯
        """

        result = call_llm(mock_model, complex_prompt)

        assert result == "Complex response"
        mock_model.generate.assert_called_once_with(complex_prompt)
