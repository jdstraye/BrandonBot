"""
Tests for Meme Detector

Tests the meme detection system to ensure it correctly identifies
political memes while NOT triggering on simple greetings or
non-political cultural references.

Borderline cases are critical - we need to distinguish:
- "Hi Brandon" (NOT meme) vs "Let's go Brandon" (IS meme)
- "Okay, Boomer" (NOT meme - general) vs "Okay, Groomer" (IS meme - political)
"""

import pytest
import asyncio
import re

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


class TestGreetingFilter:
    """Test the greeting pre-filter to ensure simple greetings skip meme detection."""
    
    @pytest.fixture
    def detector(self):
        from meme_detector import MemeDetector
        return MemeDetector()
    
    def test_hi_brandon_is_greeting(self, detector):
        """'Hi Brandon' should be detected as a greeting."""
        assert detector._is_greeting("Hi Brandon") is True
    
    def test_hi_brandon_with_period(self, detector):
        """'Hi Brandon.' should be detected as a greeting."""
        assert detector._is_greeting("Hi Brandon.") is True
    
    def test_hello_brandon(self, detector):
        """'Hello Brandon' should be detected as a greeting."""
        assert detector._is_greeting("Hello Brandon") is True
    
    def test_hey_brandon(self, detector):
        """'Hey Brandon' should be detected as a greeting."""
        assert detector._is_greeting("Hey Brandon") is True
    
    def test_hi_brandon_im_name(self, detector):
        """'Hi Brandon, I'm Jayson' should be detected as a greeting."""
        assert detector._is_greeting("Hi Brandon, I'm Jayson") is True
        assert detector._is_greeting("Hi Brandon, I'm Jayson.") is True
    
    def test_hi_brandon_how_are_you(self, detector):
        """'Hi Brandon, how are you?' should be detected as a greeting."""
        assert detector._is_greeting("Hi Brandon, how are you?") is True
        assert detector._is_greeting("Hi Brandon, how are you today?") is True
    
    def test_good_morning_brandon(self, detector):
        """'Good morning Brandon' should be detected as a greeting."""
        assert detector._is_greeting("Good morning Brandon") is True
        assert detector._is_greeting("Good afternoon Brandon") is True
        assert detector._is_greeting("Good evening Brandon") is True
    
    def test_lets_go_brandon_not_greeting(self, detector):
        """'Let's go Brandon' should NOT be a greeting - it's a meme."""
        assert detector._is_greeting("Let's go Brandon") is False
        assert detector._is_greeting("Lets go Brandon!") is False
    
    def test_hi_brandon_lets_go_not_greeting(self, detector):
        """'Hi Brandon, let's go!' should NOT be a greeting - contains meme phrase."""
        assert detector._is_greeting("Hi Brandon, let's go!") is False
    
    def test_policy_question_not_greeting(self, detector):
        """Policy questions should not be greetings."""
        assert detector._is_greeting("What is your position on immigration?") is False
        assert detector._is_greeting("Build the wall") is False
    
    def test_multi_word_names(self, detector):
        """'Hi Brandon, I'm Mary Ann' should be detected as a greeting."""
        assert detector._is_greeting("Hi Brandon, I'm Mary Ann") is True
        assert detector._is_greeting("Hi Brandon, I'm O'Connor") is True
        assert detector._is_greeting("Hello Brandon, my name is Jean-Pierre") is True
    
    def test_punctuation_variants(self, detector):
        """Various punctuation should still be greetings."""
        assert detector._is_greeting("Hi Brandon – how's it going?") is True
        assert detector._is_greeting("Hi Brandon - what's up?") is True
        assert detector._is_greeting("Hey Brandon, nice to meet you!") is True
    
    def test_meme_trigger_not_greeting(self, detector):
        """Greetings with meme triggers should NOT be detected as greeting."""
        assert detector._is_greeting("Hi Brandon, let's go!") is False
        assert detector._is_greeting("Hello, build the wall!") is False
        assert detector._is_greeting("Hey Brandon, what is a woman?") is False


class TestMemeDetectionResults:
    """Test the full meme detection flow with web search."""
    
    @pytest.fixture
    def detector(self):
        from meme_detector import MemeDetector
        return MemeDetector()
    
    def test_greeting_skips_detection(self, detector):
        """Greetings should skip detection entirely."""
        async def run():
            return await detector.detect("Hi Brandon")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False
        assert "greeting" in result.reasoning.lower()
    
    def test_greeting_with_name_skips_detection(self, detector):
        """Greetings with names should skip detection."""
        async def run():
            return await detector.detect("Hi Brandon, I'm Jayson.")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False
        assert "greeting" in result.reasoning.lower()


class TestBorderlineCases:
    """
    Test borderline cases that distinguish political memes from
    general cultural references or innocent phrases.
    
    These are the tricky cases where we need clear delineation.
    """
    
    @pytest.fixture
    def detector(self):
        from meme_detector import MemeDetector
        return MemeDetector()
    
    def test_what_is_a_tree_not_meme(self, detector):
        """'What is a tree?' is NOT a meme - just a question."""
        async def run():
            return await detector.detect("What is a tree?")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False, f"'What is a tree?' should not be a meme, got: {result.reasoning}"
    
    def test_what_is_a_man_not_meme(self, detector):
        """'What is a man?' is NOT a meme - unlike 'What is a woman?'."""
        async def run():
            return await detector.detect("What is a man?")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False, f"'What is a man?' should not be a meme, got: {result.reasoning}"
    
    def test_lets_go_alone_not_meme(self, detector):
        """'Let's Go!' alone is NOT a meme without 'Brandon'."""
        async def run():
            return await detector.detect("Let's Go!")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False, f"'Let's Go!' should not be a meme, got: {result.reasoning}"
    
    def test_build_the_team_not_meme(self, detector):
        """'Build the Team' is NOT a meme - unlike 'Build the Wall'."""
        async def run():
            return await detector.detect("Build the Team")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False, f"'Build the Team' should not be a meme, got: {result.reasoning}"
    
    def test_okay_boomer_not_meme(self, detector):
        """'Okay, Boomer' is a general cultural meme, NOT politically relevant."""
        async def run():
            return await detector.detect("Okay, Boomer")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False, f"'Okay, Boomer' should not trigger political meme detection, got: {result.reasoning}"
    
    def test_this_is_sparta_not_meme(self, detector):
        """'This is sparta!' is a movie meme, NOT politically relevant."""
        async def run():
            return await detector.detect("This is sparta!")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is False, f"'This is sparta!' should not be a meme, got: {result.reasoning}"


class TestPoliticalMemes:
    """
    Test cases that SHOULD be detected as political memes.
    These have clear political/controversy context.
    
    NOTE: These tests require web search to work and may be slow.
    They are marked with pytest.mark.slow for optional skipping.
    """
    
    @pytest.fixture
    def detector(self):
        from meme_detector import MemeDetector
        return MemeDetector()
    
    @pytest.mark.slow
    def test_lets_go_brandon_is_meme(self, detector):
        """'Let's go Brandon' IS the canonical political meme."""
        async def run():
            return await detector.detect("Let's go Brandon")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'Let's go Brandon' should be detected as meme, got: {result.reasoning}"
    
    @pytest.mark.slow
    def test_build_the_wall_is_meme(self, detector):
        """'Build the Wall' IS a political meme/slogan."""
        async def run():
            return await detector.detect("Build the Wall")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'Build the Wall' should be detected as meme, got: {result.reasoning}"
    
    @pytest.mark.slow
    def test_covfefe_is_meme(self, detector):
        """'Covfefe' IS a political meme."""
        async def run():
            return await detector.detect("Covfefe")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'Covfefe' should be detected as meme, got: {result.reasoning}"
    
    @pytest.mark.slow
    def test_okay_groomer_is_meme(self, detector):
        """'Okay, Groomer' IS a political meme - unlike 'Okay, Boomer'."""
        async def run():
            return await detector.detect("Okay, Groomer")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'Okay, Groomer' should be detected as meme, got: {result.reasoning}"
    
    @pytest.mark.slow
    def test_dark_brandon_is_meme(self, detector):
        """'Dark Brandon' IS a political meme."""
        async def run():
            return await detector.detect("Dark Brandon")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'Dark Brandon' should be detected as meme, got: {result.reasoning}"
    
    @pytest.mark.slow
    def test_this_is_fine_is_meme(self, detector):
        """'This is fine' IS a political/cultural meme (dog in fire)."""
        async def run():
            return await detector.detect("This is fine")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'This is fine' should be detected as meme, got: {result.reasoning}"
    
    @pytest.mark.slow
    def test_what_is_a_woman_is_meme(self, detector):
        """'What is a woman?' IS a political meme (Matt Walsh documentary)."""
        async def run():
            return await detector.detect("What is a woman?")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.is_meme is True, f"'What is a woman?' should be detected as meme, got: {result.reasoning}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
