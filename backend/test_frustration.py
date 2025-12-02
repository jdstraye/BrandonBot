#!/usr/bin/env python3
"""
Test Frustration Detection

Tests the sentiment-based frustration classification:
1. Pattern flags (profanity, insults, all_caps, etc.)
2. Sentiment analysis score (negative/positive/neutral)
3. Combined decision (ESCALATE vs CONTINUE)

Measures latency and accuracy.
"""

import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_CASES = {
    "calm_clear": [
        ("What is Brandon's position on healthcare?", False),
        ("Can you explain the tax policy?", False),
        ("I'd like to learn about education reform.", False),
        ("What are your thoughts on immigration?", False),
        ("How will you address the economy?", False),
    ],
    "calm_curious": [
        ("I'm curious about your environmental policy.", False),
        ("Could you tell me more about border security?", False),
        ("What's your stance on the second amendment?", False),
        ("I have a question about job creation.", False),
        ("How do you plan to reduce spending?", False),
    ],
    "mild_concern": [
        ("I'm a bit worried about the direction things are going.", False),
        ("This is concerning, but I'd like to understand better.", False),
        ("I'm not sure I agree, can you explain more?", False),
        ("I've heard conflicting things about this.", False),
        ("I'm skeptical but willing to listen.", False),
    ],
    "frustrated": [
        ("I'm so tired of politicians not listening!", True),
        ("Why can't anyone give a straight answer??", True),
        ("This is ridiculous! Nothing ever changes.", True),
        ("I've asked this question a hundred times!", True),
        ("Politicians are all the same, useless!", True),
    ],
    "angry": [
        ("You're all corrupt liars!", True),
        ("This is complete bullshit!", True),
        ("I'm sick of this garbage!", True),
        ("What the hell is wrong with you people?!", True),
        ("You're fucking useless, just like the rest!", True),
    ],
    "demands_human": [
        ("I need to talk to a real person NOW!", True),
        ("Get me to someone who can actually help!", True),
        ("I want to speak to your supervisor!", True),
        ("Transfer me to a human immediately!", True),
        ("Stop wasting my time, connect me to someone real!", True),
    ],
    "all_caps": [
        ("WHY WON'T ANYONE ANSWER MY QUESTION??", True),
        ("THIS IS UNACCEPTABLE BEHAVIOR!", True),
        ("I DEMAND AN EXPLANATION RIGHT NOW!", True),
        ("ANSWER THE DAMN QUESTION!", True),
        ("STOP IGNORING THE PEOPLE!", True),
    ],
    "repeated_punct": [
        ("Are you even listening to me???", True),
        ("Hello?!?! Is anyone there???", True),
        ("Why won't you answer!!!!", True),
        ("This is ridiculous!!!???", True),
        ("What's going on here?!?!?!", True),
    ],
    "mixed_signals": [
        ("I appreciate your work but I'm getting frustrated.", False),
        ("Thanks for trying, but this isn't working.", False),
        ("I understand it's complicated, but please help.", False),
        ("I'm not angry, just disappointed.", False),
        ("I know you're trying, but I need real answers.", False),
    ],
    "sarcastic": [
        ("Oh great, another politician dodging questions.", True),
        ("Wow, what a surprise, no real answer.", True),
        ("Sure, I'll just wait forever for a response.", True),
        ("Yeah right, like that's ever going to happen.", True),
        ("Oh wonderful, more empty promises.", True),
    ],
    "profanity_mild": [
        ("What the hell is going on with taxes?", True),
        ("This damn policy makes no sense!", True),
        ("I don't give a damn about excuses!", True),
        ("Why the hell would anyone support this?", True),
        ("This is a crap response!", True),
    ],
    "passive_aggressive": [
        ("I guess expecting competence is too much to ask.", True),
        ("Must be nice to ignore voter concerns.", True),
        ("I'm sure you'll just dodge this like everything else.", True),
        ("How convenient that you never answer tough questions.", True),
        ("I suppose actually helping people isn't a priority.", True),
    ],
}

async def test_frustration_detection():
    """Test frustration detection with latency measurement."""
    
    print("=" * 60)
    print("Frustration Detection Test")
    print("=" * 60)
    
    print("\n[1/2] Loading Sentiment Model...")
    load_start = time.time()
    
    from slm_manager import SLMManager
    from prequalifier import PatternFlags
    
    slm = SLMManager()
    await slm._ensure_sentiment_loaded()
    load_time = time.time() - load_start
    print(f"Sentiment model loaded in {load_time:.2f}s")
    
    print("\n[2/2] Running frustration classification tests...")
    print("-" * 60)
    
    results = {}
    for category in TEST_CASES.keys():
        results[category] = {"correct": 0, "total": 0, "latencies": []}
    
    for category, test_cases in TEST_CASES.items():
        print(f"\n### {category.upper()} ###")
        
        for message, should_escalate in test_cases:
            start_time = time.time()
            
            flags = PatternFlags()
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['fuck', 'shit', 'bastard', 'asshole', 'bitch']):
                flags.profanity = True
            elif any(word in message_lower for word in ['damn', 'hell', 'crap']):
                flags.profanity = True
            
            if any(word in message_lower for word in ['useless', 'idiot', 'stupid', 'moron', 'corrupt', 'liar']):
                flags.insults = True
            
            if any(phrase in message_lower for phrase in ['talk to a real person', 'speak to someone', 'supervisor', 'transfer me', 'human']):
                flags.demands_human = True
            
            words = message.split()
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            if caps_words >= len(words) * 0.5 and len(words) >= 3:
                flags.caps = True
            
            if message.count('?') >= 2 or message.count('!') >= 2:
                flags.repeated_punct = True
            
            response = await slm.classify_frustration(message, flags.to_dict())
            
            total_time = time.time() - start_time
            
            predicted_escalate = response.decision == "ESCALATE"
            is_correct = predicted_escalate == should_escalate
            
            results[category]["total"] += 1
            results[category]["latencies"].append(total_time)
            if is_correct:
                results[category]["correct"] += 1
            
            expected = "ESCALATE" if should_escalate else "CONTINUE"
            status = "PASS" if is_correct else "FAIL"
            
            print(f"  [{status}] \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            print(f"        -> {response.decision} (expected: {expected}, conf: {response.confidence:.2f}, time: {total_time*1000:.0f}ms)")
            if not is_correct:
                print(f"        Explanation: {response.explanation}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_correct = 0
    total_tests = 0
    all_latencies = []
    
    for category, data in results.items():
        accuracy = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        avg_latency = sum(data["latencies"]) / len(data["latencies"]) * 1000 if data["latencies"] else 0
        
        print(f"{category:20s}: {data['correct']}/{data['total']} ({accuracy:.0f}%) - avg latency: {avg_latency:.0f}ms")
        
        total_correct += data["correct"]
        total_tests += data["total"]
        all_latencies.extend(data["latencies"])
    
    print("-" * 60)
    overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
    avg_total_latency = sum(all_latencies) / len(all_latencies) * 1000 if all_latencies else 0
    
    print(f"{'OVERALL':20s}: {total_correct}/{total_tests} ({overall_accuracy:.0f}%)")
    print(f"{'AVG LATENCY':20s}: {avg_total_latency:.0f}ms per query")
    print(f"{'MODEL LOAD TIME':20s}: {load_time:.2f}s (one-time)")
    
    print("\n" + "=" * 60)
    if overall_accuracy >= 80:
        print("RESULT: Frustration detection is VIABLE")
    elif overall_accuracy >= 60:
        print("RESULT: Frustration detection needs TUNING")
    else:
        print("RESULT: Frustration detection needs MAJOR WORK")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_frustration_detection())
