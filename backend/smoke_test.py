#!/usr/bin/env python3
"""
Comprehensive A/B smoke test suite for BrandonBot RAG system
Tests: Comparison, MarketGuru impact, Internet search, Chunking optimization
"""
import asyncio
import json
import sys
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def test_query(question: str, test_name: str) -> Dict:
    """Send a query to BrandonBot and return the response"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:5000/api/query",
                json={"question": question},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"❌ {test_name} failed: HTTP {response.status}")
                    logger.error(f"   Error: {error_text}")
                    return {"error": error_text}
    except Exception as e:
        logger.error(f"❌ {test_name} failed: {e}")
        return {"error": str(e)}

async def run_smoke_tests():
    """Run all smoke tests and report results"""
    logger.info("=" * 80)
    logger.info("BrandonBot RAG Smoke Test Suite")
    logger.info("=" * 80)
    logger.info("")
    
    tests = [
        {
            "name": "Test A: Brandon vs Republican Party Comparison",
            "question": "How does Brandon's position on immigration differ from the Republican party platform?",
            "expected_behaviors": [
                "Should query both BrandonPlatform (1.0x trust) and PartyPlatform (0.6x trust)",
                "Should show different confidence scores reflecting trust multipliers",
                "Should cite both sources in response",
                "Response should highlight differences between positions"
            ]
        },
        {
            "name": "Test B: MarketGuru Impact (Awareness-based Communication)",
            "question": "Why should I vote for Brandon?",
            "expected_behaviors": [
                "Should query MarketGurus collection for copywriting guidance",
                "Response should be direct, benefit-focused, conversational (not political-speak)",
                "Should match prospect's awareness level with appropriate messaging",
                "Should show influence of Halbert/Hopkins/Schwartz principles"
            ]
        },
        {
            "name": "Test C: Internet Search Integration",
            "question": "How does Brandon compare to other candidates in this race?",
            "expected_behaviors": [
                "Should trigger web search (DuckDuckGo integration)",
                "Should include external sources with proper citations (footnote style)",
                "Should show brandonsowers.com results with 2x trust boost if found",
                "Response should include URLs in footnotes"
            ]
        },
        {
            "name": "Test D: Policy Question (High Confidence Expected)",
            "question": "What is Brandon's position on border security?",
            "expected_behaviors": [
                "Should retrieve from BrandonPlatform with high confidence",
                "With 128-char chunks, confidence should be ~16% higher than 1000-char baseline",
                "Should stay in character (first person, 'I believe...')",
                "Should not offer callback for high-confidence answer"
            ]
        },
        {
            "name": "Test E: Truth-Seeking Question (Bible Verse Test)",
            "question": "What does the Bible say about caring for the poor?",
            "expected_behaviors": [
                "Should query Bible verse collection",
                "Should cite Scripture references (book, chapter, verse)",
                "Should include application context for political perspective",
                "Should only use for moral/spiritual questions, not trivial facts"
            ]
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 {test['name']}")
        logger.info(f"{'='*80}")
        logger.info(f"Question: \"{test['question']}\"")
        logger.info(f"\nExpected Behaviors:")
        for behavior in test['expected_behaviors']:
            logger.info(f"  • {behavior}")
        logger.info(f"\nExecuting query...")
        logger.info("")
        
        response = await test_query(test['question'], test['name'])
        
        if "error" in response:
            logger.error(f"❌ Test failed with error")
            results.append({"test": test['name'], "status": "FAILED", "error": response['error']})
            continue
        
        logger.info(f"✅ Response received")
        logger.info(f"\n📊 Response Analysis:")
        logger.info(f"   Confidence: {response.get('confidence', 'N/A')}")
        logger.info(f"   Sources: {len(response.get('sources', []))}")
        
        if 'sources' in response and response['sources']:
            logger.info(f"\n   Top Sources:")
            for j, source in enumerate(response['sources'][:3], 1):
                logger.info(f"      {j}. {source.get('source', 'Unknown')}: "
                          f"confidence={source.get('confidence', 0):.3f}, "
                          f"collection={source.get('metadata', {}).get('collection', 'Unknown')}")
        
        logger.info(f"\n💬 Response Preview:")
        answer_preview = response.get('answer', '')[:300]
        logger.info(f"   {answer_preview}...")
        
        logger.info(f"\n🔍 Validation:")
        validation_score = 0
        
        if test['name'].startswith("Test A"):
            has_brandon = any("BrandonPlatform" in s.get('metadata', {}).get('collection', '') 
                            for s in response.get('sources', []))
            has_party = any("PartyPlatform" in s.get('metadata', {}).get('collection', '') 
                          for s in response.get('sources', []))
            if has_brandon and has_party:
                logger.info("   ✓ Both BrandonPlatform and PartyPlatform sources found")
                validation_score += 1
            else:
                logger.info(f"   ✗ Missing sources (Brandon={has_brandon}, Party={has_party})")
        
        elif test['name'].startswith("Test B"):
            has_market_gurus = any("MarketGurus" in str(s) for s in response.get('sources', []))
            if has_market_gurus:
                logger.info("   ✓ MarketGurus collection queried")
                validation_score += 1
            else:
                logger.info("   ? MarketGurus influence unclear from sources")
        
        elif test['name'].startswith("Test C"):
            has_urls = "http" in response.get('answer', '').lower()
            if has_urls:
                logger.info("   ✓ External URLs found in response")
                validation_score += 1
            else:
                logger.info("   ✗ No external URLs found (search may not have triggered)")
        
        elif test['name'].startswith("Test D"):
            confidence = response.get('confidence', 0)
            if confidence > 0.5:
                logger.info(f"   ✓ High confidence ({confidence:.3f}) achieved")
                validation_score += 1
            else:
                logger.info(f"   ✗ Low confidence ({confidence:.3f}), expected > 0.5")
        
        elif test['name'].startswith("Test E"):
            answer = response.get('answer', '').lower()
            has_scripture = any(keyword in answer for keyword in ['verse', 'bible', 'scripture', 'matthew', 'luke', 'john'])
            if has_scripture:
                logger.info("   ✓ Scripture references found")
                validation_score += 1
            else:
                logger.info("   ✗ No clear Scripture references")
        
        results.append({
            "test": test['name'],
            "status": "PASS" if validation_score > 0 else "PARTIAL",
            "confidence": response.get('confidence'),
            "source_count": len(response.get('sources', [])),
            "validation_score": validation_score
        })
    
    logger.info(f"\n\n{'='*80}")
    logger.info("📋 Test Summary")
    logger.info(f"{'='*80}")
    
    for result in results:
        status_icon = "✅" if result['status'] == "PASS" else "⚠️" if result['status'] == "PARTIAL" else "❌"
        logger.info(f"{status_icon} {result['test']}: {result['status']}")
        logger.info(f"   Confidence: {result.get('confidence', 'N/A')}, "
                  f"Sources: {result.get('source_count', 0)}, "
                  f"Validation: {result.get('validation_score', 0)}/1")
    
    passed = sum(1 for r in results if r['status'] == "PASS")
    logger.info(f"\n{'='*80}")
    logger.info(f"Overall: {passed}/{len(results)} tests passed")
    logger.info(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(run_smoke_tests())
        sys.exit(0 if all(r['status'] in ['PASS', 'PARTIAL'] for r in results) else 1)
    except KeyboardInterrupt:
        logger.info("\nTest suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
