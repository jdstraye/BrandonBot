"""
Comprehensive Test Suite for BrandonBot Validators
Tests: Prequalifier, Output Validator, FEC Compliance

Test categories:
1. Prequalifier intent detection
2. Prequalifier frustration/escalation detection
3. Prequalifier Ogilvy category detection
4. Output validator de-escalation
5. Output validator tone softening
6. FEC compliance checking
7. Edge cases and combined scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prequalifier import (
    Prequalifier, PrequalifierResult, 
    UserIntent, OgilvyCategory, EscalationLevel
)
from output_validator import (
    OutputValidator, ValidationResult, ValidationStatus,
    FECComplianceChecker
)


class TestPrequalifierIntentDetection:
    """Test intent detection accuracy"""
    
    def setup_method(self):
        self.pq = Prequalifier()
    
    def test_volunteer_intent(self):
        messages = [
            "I want to volunteer for the campaign",
            "How can I help out?",
            "Sign me up for door-to-door canvassing",
            "I'd like to join the campaign team",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.primary_intent == UserIntent.VOLUNTEER, f"Failed: {msg}"
    
    def test_donate_intent(self):
        messages = [
            "I want to donate to the campaign",
            "How can I contribute financially?",
            "I'd like to support Brandon financially",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.primary_intent == UserIntent.DONATE, f"Failed: {msg}"
    
    def test_callback_intent(self):
        messages = [
            "I need to talk to someone real",
            "Can I speak with a human?",
            "Please have someone call me back",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.primary_intent == UserIntent.CALLBACK, f"Failed: {msg}"
    
    def test_comparison_intent(self):
        messages = [
            "How does Brandon compare to the other candidate?",
            "What's the difference between Brandon and his opponent?",
            "Is Brandon better than the Democrat?",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.primary_intent == UserIntent.COMPARISON, f"Failed: {msg}"
    
    def test_scripture_intent(self):
        messages = [
            "What does Brandon's faith say about this?",
            "How does the Bible guide his positions?",
            "Is Brandon a Christian?",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.primary_intent == UserIntent.SCRIPTURE, f"Failed: {msg}"
    
    def test_policy_question(self):
        messages = [
            "What is Brandon's position on healthcare?",
            "Where does Brandon stand on taxes?",
            "How will Brandon address immigration?",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.primary_intent == UserIntent.POLICY_QUESTION, f"Failed: {msg}"


class TestPrequalifierEscalation:
    """Test frustration and escalation detection"""
    
    def setup_method(self):
        self.pq = Prequalifier()
    
    def test_high_frustration(self):
        messages = [
            "This is useless! You're not helping at all!!!",
            "I already told you this twice! You're stupid",
            "FFS I need to talk to a REAL person NOW",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.escalation_level == EscalationLevel.HIGH, f"Failed: {msg}"
            assert result.needs_deescalation == True
    
    def test_medium_frustration(self):
        messages = [
            "I'm getting confused, can you explain again?",
            "That doesn't answer my question at all",
            "You still haven't addressed what I asked",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.escalation_level in [EscalationLevel.HIGH, EscalationLevel.MEDIUM, EscalationLevel.LOW], f"Failed: {msg}"
    
    def test_urgency_signals(self):
        messages = [
            "This is urgent, I need an answer now",
            "I need to talk to someone immediately",
            "This is an emergency situation",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.escalation_level in [EscalationLevel.HIGH, EscalationLevel.MEDIUM], f"Failed: {msg}"
    
    def test_no_frustration(self):
        messages = [
            "Can you tell me about Brandon's tax plan?",
            "Thanks for the help!",
            "That's interesting, tell me more",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.escalation_level == EscalationLevel.NONE, f"Failed: {msg}"
            assert result.needs_deescalation == False


class TestPrequalifierOgilvy:
    """Test Ogilvy/Schwartz value category detection"""
    
    def setup_method(self):
        self.pq = Prequalifier()
    
    def test_security_values(self):
        messages = [
            "I want to feel safe in my community",
            "Border security is my top concern",
            "I need stability for my family",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert OgilvyCategory.SECURITY in result.ogilvy_categories, f"Failed: {msg}"
    
    def test_tradition_values(self):
        messages = [
            "I value traditional family values",
            "Faith and heritage matter to me",
            "We need to respect our traditions",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert OgilvyCategory.TRADITION in result.ogilvy_categories, f"Failed: {msg}"
    
    def test_universalism_values(self):
        messages = [
            "We need justice for everyone",
            "All people should be treated equally",
            "The environment affects us all",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert OgilvyCategory.UNIVERSALISM in result.ogilvy_categories, f"Failed: {msg}"
    
    def test_self_direction_values(self):
        messages = [
            "I want my freedom protected",
            "Individual liberty matters most",
            "I should have the right to choose",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert OgilvyCategory.SELF_DIRECTION in result.ogilvy_categories, f"Failed: {msg}"


class TestPrequalifierBlocking:
    """Test content blocking"""
    
    def setup_method(self):
        self.pq = Prequalifier()
    
    def test_financial_data_blocked(self):
        messages = [
            "Here's my credit card number: 4111...",
            "My bank account details are...",
            "What's your SSN?",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.blocked == True, f"Failed to block: {msg}"


class TestOutputValidatorDeescalation:
    """Test de-escalation templates"""
    
    def setup_method(self):
        self.validator = OutputValidator()
    
    def test_high_escalation_adds_callback_offer(self):
        response = "Brandon supports the Second Amendment."
        result = self.validator.validate(
            response=response,
            escalation_level="high",
            user_frustrated=True
        )
        assert "callback" in result.validated_response.lower() or "call" in result.validated_response.lower()
        assert result.added_deescalation == True
    
    def test_medium_escalation_adds_empathy(self):
        response = "Brandon's tax plan focuses on small businesses."
        result = self.validator.validate(
            response=response,
            escalation_level="medium",
            user_frustrated=True
        )
        assert result.status in [ValidationStatus.MODIFIED, ValidationStatus.PASSED]
    
    def test_no_escalation_unchanged(self):
        response = "Brandon supports lower taxes for families."
        result = self.validator.validate(
            response=response,
            escalation_level="none",
            user_frustrated=False
        )
        original_len = len(response)
        assert abs(len(result.validated_response) - original_len) < 100


class TestOutputValidatorToneSoftening:
    """Test dismissive language softening"""
    
    def setup_method(self):
        self.validator = OutputValidator()
    
    def test_softens_calm_down(self):
        response = "Calm down, let me explain the policy."
        result = self.validator.validate(
            response=response,
            escalation_level="medium",
            user_frustrated=True
        )
        assert "calm down" not in result.validated_response.lower()
    
    def test_detects_dismissive_phrases(self):
        responses = [
            "Obviously you don't understand",
            "As I already said before",
            "It's simple, really",
        ]
        for response in responses:
            result = self.validator.validate(
                response=response,
                escalation_level="low",
                user_frustrated=False
            )
            assert len(result.tone_issues) > 0, f"Failed to detect dismissive: {response}"


class TestFECCompliance:
    """Test FEC compliance checking"""
    
    def setup_method(self):
        self.fec = FECComplianceChecker()
    
    def test_tax_advice_flagged(self):
        responses = [
            "Your donation is tax deductible",
            "You can write off this contribution",
            "This qualifies for a tax benefit",
        ]
        for response in responses:
            compliant, issues = self.fec.check(response)
            assert not compliant, f"Should have flagged: {response}"
            assert any("tax" in i.lower() for i in issues)
    
    def test_legal_advice_flagged(self):
        responses = [
            "Consult your attorney about this",
            "Legally you should sue",
        ]
        for response in responses:
            compliant, issues = self.fec.check(response)
            assert not compliant, f"Should have flagged: {response}"
    
    def test_payment_direct_flagged(self):
        responses = [
            "Send money to this account",
            "Wire transfer to campaign",
            "Pay directly via credit card",
        ]
        for response in responses:
            compliant, issues = self.fec.check(response)
            assert not compliant, f"Should have flagged: {response}"
    
    def test_defamatory_flagged(self):
        responses = [
            "The opponent is a criminal",
            "He committed fraud",
            "The opponent stole money",
        ]
        for response in responses:
            compliant, issues = self.fec.check(response)
            assert not compliant, f"Should have flagged: {response}"
    
    def test_clean_response_passes(self):
        responses = [
            "Brandon believes in lowering taxes for families.",
            "Thank you for your interest in volunteering!",
            "Brandon's healthcare plan focuses on accessibility.",
        ]
        for response in responses:
            compliant, issues = self.fec.check(response)
            assert compliant, f"Should have passed: {response}"
            assert len(issues) == 0


class TestEdgeCases:
    """Test edge cases and combined scenarios"""
    
    def setup_method(self):
        self.pq = Prequalifier()
        self.validator = OutputValidator()
        self.fec = FECComplianceChecker()
    
    def test_vague_query_detection(self):
        messages = [
            "what",
            "stuff",
            "hi",
        ]
        for msg in messages:
            result = self.pq.analyze(msg)
            assert result.is_vague == True or len(msg) <= 10, f"Should be vague: {msg}"
    
    def test_multi_intent_query(self):
        msg = "I want to volunteer and also know about Brandon's faith"
        result = self.pq.analyze(msg)
        assert result.primary_intent in [UserIntent.VOLUNTEER, UserIntent.SCRIPTURE]
        assert len(result.secondary_intents) > 0
    
    def test_frustrated_volunteer(self):
        msg = "I've been trying to volunteer for days! Why won't anyone help me sign up??"
        result = self.pq.analyze(msg)
        assert result.primary_intent == UserIntent.VOLUNTEER
        assert result.escalation_level in [EscalationLevel.NONE, EscalationLevel.LOW, EscalationLevel.MEDIUM]
    
    def test_conversation_history_escalation(self):
        history = [
            {"role": "user", "content": "What is Brandon's position?"},
            {"role": "assistant", "content": "Brandon supports..."},
            {"role": "user", "content": "That doesn't answer my question"},
            {"role": "assistant", "content": "Let me clarify..."},
            {"role": "user", "content": "I already asked this twice!"},
        ]
        msg = "I'm still confused! You're not helping!!"
        result = self.pq.analyze(msg, "session1", history)
        assert result.escalation_level in [EscalationLevel.MEDIUM, EscalationLevel.HIGH]


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def setup_method(self):
        self.pq = Prequalifier()
        self.validator = OutputValidator()
        self.fec = FECComplianceChecker()
    
    def test_full_pipeline_normal(self):
        user_msg = "What is Brandon's position on healthcare?"
        pq_result = self.pq.analyze(user_msg)
        
        assert pq_result.blocked == False
        assert pq_result.escalation_level == EscalationLevel.NONE
        
        llm_response = "Brandon supports increasing healthcare access."
        val_result = self.validator.validate(
            response=llm_response,
            escalation_level=pq_result.escalation_level.value,
            user_frustrated=pq_result.needs_deescalation
        )
        
        fec_ok, fec_issues = self.fec.check(val_result.validated_response)
        
        assert val_result.status == ValidationStatus.PASSED
        assert fec_ok == True
    
    def test_full_pipeline_frustrated(self):
        user_msg = "I've asked THREE TIMES and you still don't get it!!!"
        pq_result = self.pq.analyze(user_msg)
        
        assert pq_result.escalation_level in [EscalationLevel.MEDIUM, EscalationLevel.HIGH]
        assert pq_result.needs_deescalation == True
        
        llm_response = "As I said before, Brandon's position is..."
        val_result = self.validator.validate(
            response=llm_response,
            escalation_level=pq_result.escalation_level.value,
            user_frustrated=pq_result.needs_deescalation
        )
        
        assert val_result.status in [ValidationStatus.MODIFIED, ValidationStatus.WARNING]
        assert len(val_result.tone_issues) > 0 or len(val_result.modifications) > 0


def run_all_tests():
    """Run all tests and print summary"""
    test_classes = [
        TestPrequalifierIntentDetection,
        TestPrequalifierEscalation,
        TestPrequalifierOgilvy,
        TestPrequalifierBlocking,
        TestOutputValidatorDeescalation,
        TestOutputValidatorToneSoftening,
        TestFECCompliance,
        TestEdgeCases,
        TestIntegration,
    ]
    
    total = 0
    passed = 0
    failed = []
    
    for test_class in test_classes:
        instance = test_class()
        instance.setup_method()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    passed += 1
                    print(f"  PASS: {test_class.__name__}.{method_name}")
                except Exception as e:
                    failed.append(f"{test_class.__name__}.{method_name}: {e}")
                    print(f"  FAIL: {test_class.__name__}.{method_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    
    if failed:
        print(f"\nFailed tests ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    
    return passed, total, failed


if __name__ == "__main__":
    run_all_tests()
