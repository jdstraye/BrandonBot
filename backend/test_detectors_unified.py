#!/usr/bin/env python3
"""
Unified Test Suite for Vagueness and Frustration Detectors

Tests both detectors against:
1. Original test cases (30 vagueness, 60 frustration)
2. 200 Arizona political phrases with vagueness/frustration scores

Measures accuracy, latency, and provides detailed results.
"""

import asyncio
import time
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARIZONA_PHRASES = [
    ("We need to secure the border now!", 2, 5),
    ("What is your stance on the Colorado River water allocation?", 1, 1),
    ("This administration is failing Arizonans.", 3, 4),
    ("Can you explain your plan for improving rural healthcare access?", 1, 1),
    ("Something must be done about the rising cost of living.", 4, 3),
    ("The election was stolen, and everyone knows it.", 2, 5),
    ("How do you plan to address the teacher shortage in Arizona?", 1, 1),
    ("Politicians never listen to the people.", 3, 4),
    ("What specific policies do you support for renewable energy in Arizona?", 1, 1),
    ("The government is out of control.", 5, 5),
    ("Can you clarify your position on Proposition 208?", 1, 1),
    ("Arizona deserves better leadership.", 3, 3),
    ("Why are property taxes so high in Maricopa County?", 2, 3),
    ("It's time to drain the swamp in Phoenix.", 4, 5),
    ("What are your thoughts on the drought contingency plan?", 1, 1),
    ("No one is doing anything about the homelessness crisis.", 3, 4),
    ("We need more transparency in government spending.", 2, 2),
    ("The media is lying to us.", 5, 5),
    ("How will you support small businesses in Tucson?", 1, 1),
    ("This is a disgrace to our state.", 5, 5),
    ("What steps are you taking to protect our water rights?", 1, 1),
    ("I'm tired of empty promises.", 4, 4),
    ("The legislature needs to focus on real issues.", 3, 3),
    ("Why is Arizona last in education funding?", 2, 4),
    ("We need to hold corrupt officials accountable.", 3, 5),
    ("Can you outline your plan for infrastructure improvements?", 1, 1),
    ("The system is rigged against everyday people.", 4, 4),
    ("What is your view on the legalization of recreational marijuana?", 1, 1),
    ("Arizona is being overrun by illegal immigrants.", 2, 5),
    ("How do you plan to lower healthcare costs?", 1, 1),
    ("The government doesn't care about us.", 5, 5),
    ("What are your priorities for the next legislative session?", 1, 1),
    ("This is unacceptable!", 5, 5),
    ("Can you explain your voting record on SB 1070?", 1, 1),
    ("Arizona's future is at stake.", 4, 3),
    ("Why are our roads in such terrible condition?", 2, 3),
    ("We need to stop the radical agenda in our schools.", 3, 4),
    ("What is your plan for economic growth in rural Arizona?", 1, 1),
    ("The elite are destroying our state.", 5, 5),
    ("How will you address the opioid crisis?", 1, 1),
    ("I'm fed up with the lack of action.", 4, 4),
    ("The border crisis is out of control.", 2, 5),
    ("What are your thoughts on the Navajo Nation's water rights?", 1, 1),
    ("No one is fighting for the middle class.", 3, 4),
    ("Can you provide details on your tax reform proposal?", 1, 1),
    ("Arizona is becoming unrecognizable.", 4, 4),
    ("Why are utility bills so high?", 2, 3),
    ("We need to restore law and order.", 3, 4),
    ("What is your position on charter schools?", 1, 1),
    ("The government is wasting our money.", 4, 4),
    ("How do you plan to support veterans in Arizona?", 1, 1),
    ("This is not the Arizona I grew up in.", 4, 4),
    ("What are your thoughts on the recent Supreme Court ruling?", 1, 1),
    ("The political establishment is failing us.", 3, 5),
    ("Can you explain your stance on gun rights?", 1, 1),
    ("Arizona needs real change.", 4, 3),
    ("Why is there so much corruption in Phoenix?", 2, 4),
    ("We need to protect our Second Amendment rights.", 2, 3),
    ("What is your plan for affordable housing?", 1, 1),
    ("The system is broken.", 5, 5),
    ("How will you improve public safety in Arizona?", 1, 1),
    ("I'm sick of the lies.", 5, 5),
    ("What are your thoughts on the recent election audit?", 1, 1),
    ("Arizona deserves leaders who actually care.", 3, 3),
    ("Why are our schools underfunded?", 2, 4),
    ("We need to stand up for our values.", 3, 3),
    ("Can you clarify your position on abortion rights?", 1, 1),
    ("The government is ignoring the will of the people.", 3, 4),
    ("What is your plan for job creation?", 1, 1),
    ("This is a betrayal of Arizonans.", 4, 5),
    ("How do you plan to address climate change?", 1, 1),
    ("The political class is out of touch.", 3, 4),
    ("What are your thoughts on the recent wildfires?", 1, 1),
    ("Arizona is being sold out to the highest bidder.", 4, 5),
    ("Can you explain your stance on the minimum wage?", 1, 1),
    ("No one is addressing the real problems.", 4, 4),
    ("What is your plan for improving broadband access in rural areas?", 1, 1),
    ("The government is failing our children.", 3, 5),
    ("How will you support local farmers?", 1, 1),
    ("This is a disaster.", 5, 5),
    ("What are your thoughts on the recent tax cuts?", 1, 1),
    ("Arizona needs leaders with integrity.", 3, 3),
    ("Why are our taxes so high?", 2, 3),
    ("We need to take back our state.", 4, 5),
    ("Can you outline your plan for criminal justice reform?", 1, 1),
    ("The system is corrupt.", 5, 5),
    ("What is your position on the death penalty?", 1, 1),
    ("Arizona is being left behind.", 4, 4),
    ("How do you plan to address the housing crisis?", 1, 1),
    ("I'm tired of the excuses.", 4, 4),
    ("The government is not representing us.", 3, 5),
    ("What are your thoughts on the recent budget surplus?", 1, 1),
    ("This is not what democracy looks like.", 4, 5),
    ("Can you explain your stance on voter ID laws?", 1, 1),
    ("Arizona needs a fresh start.", 4, 3),
    ("Why are our wages so low?", 2, 4),
    ("We need to put Arizona first.", 3, 3),
    ("What is your plan for reducing crime?", 1, 1),
    ("The political elite don't care about us.", 3, 5),
    ("How will you support Arizona's tribal communities?", 1, 1),
    ("This is a shameful display of leadership.", 4, 5),
    ("What are your thoughts on the recent drought declaration?", 1, 1),
    ("Arizona deserves better.", 5, 3),
    ("Why is healthcare so expensive?", 2, 4),
    ("We need to restore trust in government.", 3, 3),
    ("Can you clarify your position on school vouchers?", 1, 1),
    ("The government is not working for us.", 4, 5),
    ("What is your plan for improving air quality?", 1, 1),
    ("This is a crisis.", 5, 5),
    ("How do you plan to address the teacher pay gap?", 1, 1),
    ("Arizona is being destroyed by bad policies.", 4, 5),
    ("What are your thoughts on the recent immigration bill?", 1, 1),
    ("No one is listening to the people.", 3, 4),
    ("Can you explain your stance on renewable energy incentives?", 1, 1),
    ("The government is failing our veterans.", 3, 5),
    ("What is your plan for improving mental health services?", 1, 1),
    ("This is a disgrace.", 5, 5),
    ("How will you address the affordable housing shortage?", 1, 1),
    ("Arizona needs real solutions.", 4, 3),
    ("Why are our roads so bad?", 2, 3),
    ("We need to hold politicians accountable.", 3, 4),
    ("What is your position on the legalization of sports betting?", 1, 1),
    ("The government is not doing enough.", 4, 4),
    ("Can you outline your plan for education reform?", 1, 1),
    ("Arizona is being mismanaged.", 4, 5),
    ("What are your thoughts on the recent water cuts?", 1, 1),
    ("I'm frustrated with the lack of progress.", 4, 4),
    ("The political system is broken.", 5, 5),
    ("How do you plan to support Arizona's economy?", 1, 1),
    ("This is not acceptable.", 5, 5),
    ("What is your stance on the recent election laws?", 1, 1),
    ("Arizona needs leaders who will fight for us.", 3, 3),
    ("Why are our schools failing?", 2, 4),
    ("We need to take action now.", 3, 4),
    ("Can you explain your position on property tax relief?", 1, 1),
    ("The government is not representing the people.", 3, 5),
    ("What is your plan for improving public transportation?", 1, 1),
    ("This is a betrayal of our values.", 4, 5),
    ("How will you address the opioid epidemic?", 1, 1),
    ("Arizona is being ignored.", 4, 4),
    ("What are your thoughts on the recent Supreme Court decision?", 1, 1),
    ("No one is fighting for Arizona.", 3, 5),
    ("Can you clarify your stance on the death penalty?", 1, 1),
    ("The government is failing our children.", 3, 5),
    ("What is your plan for reducing homelessness?", 1, 1),
    ("This is not the Arizona we know.", 4, 4),
    ("How do you plan to support small businesses?", 1, 1),
    ("The political class is out of touch with reality.", 3, 5),
    ("What are your thoughts on the recent tax hike?", 1, 1),
    ("Arizona deserves leaders who will stand up for us.", 3, 3),
    ("Why are our utilities so expensive?", 2, 3),
    ("We need to restore common sense to government.", 3, 3),
    ("Can you explain your position on the recent election audit?", 1, 1),
    ("The government is not working for the people.", 4, 5),
    ("What is your plan for improving healthcare access?", 1, 1),
    ("This is a failure of leadership.", 4, 5),
    ("How will you address the affordable housing crisis?", 1, 1),
    ("Arizona needs real change, not empty promises.", 3, 4),
    ("Why are our wages stagnant?", 2, 4),
    ("We need to put Arizona first.", 3, 3),
    ("What is your stance on the recent immigration policies?", 1, 1),
    ("The government is not listening.", 4, 5),
    ("Can you outline your plan for criminal justice reform?", 1, 1),
    ("Arizona is being let down by its leaders.", 4, 4),
    ("What are your thoughts on the recent wildfire prevention efforts?", 1, 1),
    ("This is not what we voted for.", 4, 5),
    ("How do you plan to support Arizona's tribal communities?", 1, 1),
    ("The government is failing Arizonans.", 3, 5),
    ("What is your plan for improving infrastructure?", 1, 1),
    ("I'm tired of the lack of action.", 4, 4),
    ("The political establishment is corrupt.", 3, 5),
    ("Can you explain your stance on renewable energy?", 1, 1),
    ("Arizona deserves better than this.", 4, 4),
    ("Why are our schools underfunded?", 2, 4),
    ("We need to hold our leaders accountable.", 3, 4),
    ("What is your position on the recent election integrity laws?", 1, 1),
    ("The government is not representing the people.", 4, 5),
    ("How will you address the teacher shortage?", 1, 1),
    ("This is a crisis of leadership.", 4, 5),
    ("What are your thoughts on the recent water rights ruling?", 1, 1),
    ("Arizona needs leaders who will fight for our values.", 3, 3),
    ("Why are our taxes so high?", 2, 3),
    ("We need to restore integrity to government.", 3, 3),
    ("Can you clarify your position on abortion rights?", 1, 1),
    ("The government is not working for us.", 4, 5),
    ("What is your plan for improving public safety?", 1, 1),
    ("This is a disgrace to our state.", 5, 5),
    ("How do you plan to support Arizona's economy?", 1, 1),
    ("Arizona is being sold out.", 4, 5),
    ("What are your thoughts on the recent budget cuts?", 1, 1),
    ("No one is fighting for the middle class.", 3, 4),
    ("Can you explain your stance on the minimum wage?", 1, 1),
    ("The government is failing our veterans.", 3, 5),
    ("What is your plan for reducing crime?", 1, 1),
    ("This is not acceptable.", 5, 5),
    ("How will you address the housing crisis?", 1, 1),
    ("Arizona needs real solutions, not empty rhetoric.", 3, 4),
    ("Why are our roads in such bad shape?", 2, 3),
    ("We need to hold our leaders accountable.", 3, 4),
    ("What is your position on the recent election laws?", 1, 1),
    ("The government is not representing the people.", 4, 5),
    ("Can you outline your plan for education reform?", 1, 1),
    ("Arizona is being mismanaged.", 4, 5),
    ("What are your thoughts on the recent water cuts?", 1, 1),
    ("I'm frustrated with the lack of progress.", 4, 4),
    ("The political system is broken.", 5, 5),
    ("How do you plan to support Arizona's economy?", 1, 1),
    ("This is a betrayal of our values.", 4, 5),
    ("What is your stance on the recent immigration bill?", 1, 1),
    ("No one is listening to the people.", 3, 4),
    ("Can you explain your position on renewable energy incentives?", 1, 1),
    ("The government is not doing enough for Arizona.", 4, 5),
    ("What is your plan for improving mental health services?", 1, 1),
    ("This is a failure of leadership.", 4, 5),
    ("How will you address the affordable housing shortage?", 1, 1),
    ("Arizona needs real change.", 4, 3),
    ("Why are our wages so low?", 2, 4),
    ("We need to put Arizona first.", 3, 3),
    ("What is your position on the recent tax hike?", 1, 1),
    ("The government is not working for the people.", 4, 5),
    ("Can you clarify your stance on the death penalty?", 1, 1),
    ("Arizona deserves leaders who will stand up for us.", 3, 3),
    ("Why are our utilities so expensive?", 2, 3),
    ("We need to restore common sense to government.", 3, 3),
    ("What is your plan for improving public transportation?", 1, 1),
    ("This is not the Arizona we know.", 4, 4),
    ("How do you plan to support small businesses?", 1, 1),
    ("The political class is out of touch with reality.", 3, 5),
    ("What are your thoughts on the recent Supreme Court decision?", 1, 1),
    ("Arizona is being ignored.", 4, 4),
    ("No one is fighting for Arizona.", 3, 5),
]

ORIGINAL_VAGUENESS_TESTS = {
    "clear_policy": [
        "What is Brandon's position on healthcare?",
        "How does Brandon plan to address inflation?",
        "What are Brandon's views on education?",
        "Tell me about Brandon's tax policy",
        "What is Brandon's stance on immigration?",
    ],
    "clear_specific": [
        "Does Brandon support the second amendment?",
        "What is Brandon's plan for job creation?",
        "How will Brandon reduce government spending?",
        "What does Brandon think about border security?",
        "Where does Brandon stand on energy independence?",
    ],
    "vague_greeting": ["hi", "hello", "hey there", "good morning", "what's up"],
    "vague_short": ["taxes", "healthcare", "help", "question", "info"],
    "vague_ambiguous": [
        "What about that thing?",
        "Tell me more",
        "What do you think?",
        "Can you explain?",
        "I have a question",
    ],
    "edge_cases": [
        "What about healthcare?",
        "Tell me about the economy",
        "What's Brandon's opinion?",
        "I want to know more about Brandon",
        "What are the policies?",
    ],
}

ORIGINAL_FRUSTRATION_TESTS = {
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
}


def vagueness_score_to_expected(score: int) -> str:
    """Convert vagueness score (1-5) to CLEAR/VAGUE decision"""
    return "VAGUE" if score >= 4 else "CLEAR"


def frustration_score_to_expected(score: int) -> bool:
    """Convert frustration score (1-5) to escalate decision"""
    return score >= 4


async def test_vagueness_detector():
    """Test vagueness detector with all test cases"""
    import weaviate
    from slm_manager import SLMManager
    from weaviate_manager import WeaviateManager
    from prequalifier import Prequalifier, RAGResult
    
    print("=" * 70)
    print("VAGUENESS DETECTOR TEST - UPGRADED BGE-RERANKER-V2-M3")
    print("=" * 70)
    
    print("\n[1/3] Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_local(port=8079, grpc_port=50050)
        if not client.is_ready():
            print("ERROR: Weaviate not ready")
            return None
        print("Weaviate connected")
    except Exception as e:
        print(f"ERROR: Could not connect to Weaviate: {e}")
        return None
    
    print("\n[2/3] Loading SLM (BGE Reranker V2 M3)...")
    load_start = time.time()
    
    slm = SLMManager()
    await slm._ensure_cross_encoder_loaded()
    load_time = time.time() - load_start
    print(f"Cross-encoder loaded in {load_time:.2f}s")
    print(f"Model: {slm.CROSS_ENCODER_MODEL}")
    
    weaviate_mgr = WeaviateManager()
    weaviate_mgr.client = client
    
    pq = Prequalifier()
    pq.set_slm_provider(slm)
    pq.set_weaviate_manager(weaviate_mgr)
    
    print("\n[3/3] Running vagueness tests...")
    print("-" * 70)
    
    results = {"original": {}, "arizona": {"correct": 0, "total": 0, "latencies": []}}
    
    for category in ORIGINAL_VAGUENESS_TESTS:
        results["original"][category] = {"correct": 0, "total": 0, "latencies": []}
    
    print("\n### ORIGINAL TEST CASES ###")
    for category, queries in ORIGINAL_VAGUENESS_TESTS.items():
        expected = "CLEAR" if category.startswith("clear") else "VAGUE"
        print(f"\n{category.upper()} (expected: {expected})")
        
        for query in queries:
            start = time.time()
            rag_results, avg_conf = await pq._retrieve_rag_context(query)
            rag_dicts = [r.to_dict() for r in rag_results] if rag_results else []
            response = await slm.classify_vagueness_with_rag(query, rag_dicts, avg_conf)
            latency = time.time() - start
            
            is_correct = response.decision == expected
            results["original"][category]["total"] += 1
            results["original"][category]["latencies"].append(latency)
            if is_correct:
                results["original"][category]["correct"] += 1
            
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] \"{query[:50]}\" -> {response.decision} ({latency*1000:.0f}ms)")
    
    print("\n### ARIZONA PHRASES (first 50) ###")
    for phrase, vagueness, frustration in ARIZONA_PHRASES[:50]:
        expected = vagueness_score_to_expected(vagueness)
        start = time.time()
        rag_results, avg_conf = await pq._retrieve_rag_context(phrase)
        rag_dicts = [r.to_dict() for r in rag_results] if rag_results else []
        response = await slm.classify_vagueness_with_rag(phrase, rag_dicts, avg_conf)
        latency = time.time() - start
        
        is_correct = response.decision == expected
        results["arizona"]["total"] += 1
        results["arizona"]["latencies"].append(latency)
        if is_correct:
            results["arizona"]["correct"] += 1
        
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] \"{phrase[:45]}...\" -> {response.decision} (exp: {expected}, {latency*1000:.0f}ms)")
    
    client.close()
    
    print("\n" + "=" * 70)
    print("VAGUENESS SUMMARY")
    print("=" * 70)
    
    total_correct = 0
    total_tests = 0
    all_latencies = []
    
    print("\nOriginal Test Cases:")
    for category, data in results["original"].items():
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        avg_lat = sum(data["latencies"]) / len(data["latencies"]) * 1000 if data["latencies"] else 0
        print(f"  {category:20s}: {data['correct']}/{data['total']} ({acc:.0f}%) - {avg_lat:.0f}ms")
        total_correct += data["correct"]
        total_tests += data["total"]
        all_latencies.extend(data["latencies"])
    
    print(f"\nArizona Phrases (50):")
    az = results["arizona"]
    az_acc = az["correct"] / az["total"] * 100 if az["total"] > 0 else 0
    az_lat = sum(az["latencies"]) / len(az["latencies"]) * 1000 if az["latencies"] else 0
    print(f"  {'Arizona':20s}: {az['correct']}/{az['total']} ({az_acc:.0f}%) - {az_lat:.0f}ms")
    total_correct += az["correct"]
    total_tests += az["total"]
    all_latencies.extend(az["latencies"])
    
    overall_acc = total_correct / total_tests * 100 if total_tests > 0 else 0
    avg_lat = sum(all_latencies) / len(all_latencies) * 1000 if all_latencies else 0
    
    print("-" * 70)
    print(f"OVERALL: {total_correct}/{total_tests} ({overall_acc:.0f}%) - avg latency: {avg_lat:.0f}ms")
    print(f"MODEL LOAD TIME: {load_time:.2f}s")
    print("=" * 70)
    
    return {"accuracy": overall_acc, "latency_ms": avg_lat, "load_time": load_time}


async def test_frustration_detector():
    """Test frustration detector with all test cases"""
    from slm_manager import SLMManager
    from prequalifier import PatternFlags
    
    print("\n" + "=" * 70)
    print("FRUSTRATION DETECTOR TEST - UPGRADED J-HARTMANN 7-EMOTION")
    print("=" * 70)
    
    print("\n[1/2] Loading Emotion Model...")
    load_start = time.time()
    
    slm = SLMManager()
    await slm._ensure_emotion_loaded()
    load_time = time.time() - load_start
    print(f"Emotion model loaded in {load_time:.2f}s")
    print(f"Model: {slm.EMOTION_MODEL}")
    print(f"Emotions: {slm.EMOTION_LABELS}")
    
    print("\n[2/2] Running frustration tests...")
    print("-" * 70)
    
    results = {"original": {}, "arizona": {"correct": 0, "total": 0, "latencies": [], "emotions": {}}}
    
    for category in ORIGINAL_FRUSTRATION_TESTS:
        results["original"][category] = {"correct": 0, "total": 0, "latencies": [], "emotions": {}}
    
    def get_flags(message: str) -> PatternFlags:
        flags = PatternFlags()
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['fuck', 'shit', 'bastard', 'asshole', 'bitch', 'damn', 'hell', 'crap']):
            flags.profanity = True
        if any(word in msg_lower for word in ['useless', 'idiot', 'stupid', 'moron', 'corrupt', 'liar']):
            flags.insults = True
        if any(phrase in msg_lower for phrase in ['talk to a real person', 'speak to someone', 'supervisor', 'transfer me', 'human']):
            flags.demands_human = True
        
        words = message.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        if caps_words >= len(words) * 0.5 and len(words) >= 3:
            flags.all_caps = True
        
        if message.count('?') >= 2 or message.count('!') >= 2:
            flags.repeated_punct = True
        
        return flags
    
    print("\n### ORIGINAL TEST CASES ###")
    for category, test_cases in ORIGINAL_FRUSTRATION_TESTS.items():
        print(f"\n{category.upper()}")
        
        for message, should_escalate in test_cases:
            start = time.time()
            flags = get_flags(message)
            response = await slm.classify_frustration(message, flags.to_dict())
            latency = time.time() - start
            
            predicted_escalate = response.decision == "ESCALATE"
            is_correct = predicted_escalate == should_escalate
            
            results["original"][category]["total"] += 1
            results["original"][category]["latencies"].append(latency)
            if is_correct:
                results["original"][category]["correct"] += 1
            
            emotion = response.detected_emotion
            results["original"][category]["emotions"][emotion] = results["original"][category]["emotions"].get(emotion, 0) + 1
            
            expected = "ESCALATE" if should_escalate else "CONTINUE"
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] \"{message[:40]}...\" -> {response.decision} (exp: {expected}) [{emotion}]")
    
    print("\n### ARIZONA PHRASES (first 50) ###")
    for phrase, vagueness, frustration in ARIZONA_PHRASES[:50]:
        should_escalate = frustration_score_to_expected(frustration)
        start = time.time()
        flags = get_flags(phrase)
        response = await slm.classify_frustration(phrase, flags.to_dict())
        latency = time.time() - start
        
        predicted_escalate = response.decision == "ESCALATE"
        is_correct = predicted_escalate == should_escalate
        
        results["arizona"]["total"] += 1
        results["arizona"]["latencies"].append(latency)
        if is_correct:
            results["arizona"]["correct"] += 1
        
        emotion = response.detected_emotion
        results["arizona"]["emotions"][emotion] = results["arizona"]["emotions"].get(emotion, 0) + 1
        
        expected = "ESCALATE" if should_escalate else "CONTINUE"
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] \"{phrase[:40]}...\" -> {response.decision} (exp: {expected}) [{emotion}]")
    
    print("\n" + "=" * 70)
    print("FRUSTRATION SUMMARY")
    print("=" * 70)
    
    total_correct = 0
    total_tests = 0
    all_latencies = []
    all_emotions = {}
    
    print("\nOriginal Test Cases:")
    for category, data in results["original"].items():
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        avg_lat = sum(data["latencies"]) / len(data["latencies"]) * 1000 if data["latencies"] else 0
        print(f"  {category:20s}: {data['correct']}/{data['total']} ({acc:.0f}%) - {avg_lat:.0f}ms")
        total_correct += data["correct"]
        total_tests += data["total"]
        all_latencies.extend(data["latencies"])
        for e, c in data["emotions"].items():
            all_emotions[e] = all_emotions.get(e, 0) + c
    
    print(f"\nArizona Phrases (50):")
    az = results["arizona"]
    az_acc = az["correct"] / az["total"] * 100 if az["total"] > 0 else 0
    az_lat = sum(az["latencies"]) / len(az["latencies"]) * 1000 if az["latencies"] else 0
    print(f"  {'Arizona':20s}: {az['correct']}/{az['total']} ({az_acc:.0f}%) - {az_lat:.0f}ms")
    total_correct += az["correct"]
    total_tests += az["total"]
    all_latencies.extend(az["latencies"])
    for e, c in az["emotions"].items():
        all_emotions[e] = all_emotions.get(e, 0) + c
    
    overall_acc = total_correct / total_tests * 100 if total_tests > 0 else 0
    avg_lat = sum(all_latencies) / len(all_latencies) * 1000 if all_latencies else 0
    
    print("-" * 70)
    print(f"OVERALL: {total_correct}/{total_tests} ({overall_acc:.0f}%) - avg latency: {avg_lat:.0f}ms")
    print(f"MODEL LOAD TIME: {load_time:.2f}s")
    
    print(f"\nEMOTION DISTRIBUTION:")
    for emotion, count in sorted(all_emotions.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count}")
    print("=" * 70)
    
    return {"accuracy": overall_acc, "latency_ms": avg_lat, "load_time": load_time, "emotions": all_emotions}


async def main():
    """Run both detector tests"""
    print("\n" + "=" * 70)
    print("UNIFIED DETECTOR TEST SUITE")
    print("Testing upgraded models: BGE-Reranker-v2-m3 + J-Hartmann 7-Emotion")
    print("=" * 70)
    
    vagueness_results = await test_vagueness_detector()
    frustration_results = await test_frustration_detector()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if vagueness_results:
        print(f"\nVAGUENESS DETECTOR:")
        print(f"  Accuracy: {vagueness_results['accuracy']:.1f}%")
        print(f"  Latency: {vagueness_results['latency_ms']:.0f}ms avg")
        print(f"  Load Time: {vagueness_results['load_time']:.1f}s")
    
    if frustration_results:
        print(f"\nFRUSTRATION DETECTOR:")
        print(f"  Accuracy: {frustration_results['accuracy']:.1f}%")
        print(f"  Latency: {frustration_results['latency_ms']:.0f}ms avg")
        print(f"  Load Time: {frustration_results['load_time']:.1f}s")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
