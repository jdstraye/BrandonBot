#!/bin/bash
# Comprehensive A/B smoke test suite for BrandonBot RAG system
# Tests: Comparison, MarketGuru impact, Internet search, Chunking optimization

echo "================================================================================"
echo "BrandonBot RAG Smoke Test Suite"
echo "================================================================================"
echo ""

# Test function
run_test() {
    local test_name="$1"
    local question="$2"
    local test_num="$3"
    
    echo "================================================================================"
    echo "🧪 Test $test_num: $test_name"
    echo "================================================================================"
    echo "Question: \"$question\""
    echo ""
    echo "Executing query..."
    
    # Send query and save response
    response=$(curl -s -X POST http://localhost:5000/api/query \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$question\"}" \
        --max-time 120)
    
    if [ $? -ne 0 ]; then
        echo "❌ Test failed: curl error"
        return 1
    fi
    
    # Parse response
    confidence=$(echo "$response" | python3.11 -c "import sys, json; data=json.load(sys.stdin); print(data.get('confidence', 0))" 2>/dev/null)
    source_count=$(echo "$response" | python3.11 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('sources', [])))" 2>/dev/null)
    answer=$(echo "$response" | python3.11 -c "import sys, json; data=json.load(sys.stdin); print(data.get('response', '')[:300])" 2>/dev/null)
    has_dual_sources=$(echo "$response" | python3.11 -c "import sys, json; data=json.load(sys.stdin); print(data.get('has_dual_sources', 'N/A'))" 2>/dev/null)
    offer_callback=$(echo "$response" | python3.11 -c "import sys, json; data=json.load(sys.stdin); print(data.get('offer_callback', False))" 2>/dev/null)
    
    echo "✅ Response received"
    echo ""
    echo "📊 Response Analysis:"
    echo "   Confidence: $confidence"
    echo "   Sources: $source_count"
    echo "   Has Dual Sources: $has_dual_sources"
    echo "   Offer Callback: $offer_callback"
    echo ""
    echo "💬 Response Preview:"
    echo "   $answer..."
    echo ""
    
    # Save full response to file for detailed analysis
    echo "$response" > "/tmp/test_${test_num}_response.json"
    echo "   Full response saved to: /tmp/test_${test_num}_response.json"
    echo ""
}

# Test A: Brandon vs Republican Party Comparison
run_test "Brandon vs Republican Party Comparison" \
    "How does Brandon's position on immigration differ from the Republican party platform?" \
    "A"

echo "Expected Behaviors for Test A:"
echo "  • Should query both BrandonPlatform (1.0x trust) and PartyPlatform (0.6x trust)"
echo "  • Should show different confidence scores reflecting trust multipliers"
echo "  • Should cite both sources in response"
echo "  • Response should highlight differences between positions"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_A_response.json') as f:
    data = json.load(f)
sources = data.get('sources', [])
has_brandon = any('BrandonPlatform' in str(s.get('metadata', {}).get('collection', '')) for s in sources)
has_party = any('PartyPlatform' in str(s.get('metadata', {}).get('collection', '')) for s in sources)
if has_brandon and has_party:
    print('   ✓ Both BrandonPlatform and PartyPlatform sources found')
else:
    print(f'   ✗ Missing sources (Brandon={has_brandon}, Party={has_party})')
"
echo ""

# Test B: MarketGuru Impact
run_test "MarketGuru Impact (Awareness-based Communication)" \
    "Why should I vote for Brandon?" \
    "B"

echo "Expected Behaviors for Test B:"
echo "  • Should query MarketGurus collection for copywriting guidance"
echo "  • Response should be direct, benefit-focused, conversational (not political-speak)"
echo "  • Should match prospect's awareness level with appropriate messaging"
echo "  • Should show influence of Halbert/Hopkins/Schwartz principles"
echo ""

# Test C: Internet Search Integration
run_test "Internet Search Integration" \
    "How does Brandon compare to other candidates in this race?" \
    "C"

echo "Expected Behaviors for Test C:"
echo "  • Should trigger web search (DuckDuckGo integration)"
echo "  • Should include external sources with proper citations (footnote style)"
echo "  • Should show brandonsowers.com results with 2x trust boost if found"
echo "  • Response should include URLs in footnotes"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_C_response.json') as f:
    data = json.load(f)
answer = data.get('answer', '').lower()
has_urls = 'http' in answer
if has_urls:
    print('   ✓ External URLs found in response')
else:
    print('   ✗ No external URLs found (search may not have triggered)')
"
echo ""

# Test D: Policy Question (High Confidence Expected)
run_test "Policy Question (High Confidence Expected)" \
    "What is Brandon's position on border security?" \
    "D"

echo "Expected Behaviors for Test D:"
echo "  • Should retrieve from BrandonPlatform with high confidence"
echo "  • With 128-char chunks, confidence should be ~16% higher than 1000-char baseline"
echo "  • Should stay in character (first person, 'I believe...')"
echo "  • Should not offer callback for high-confidence answer"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_D_response.json') as f:
    data = json.load(f)
confidence = data.get('confidence', 0)
if confidence > 0.5:
    print(f'   ✓ High confidence ({confidence:.3f}) achieved')
else:
    print(f'   ✗ Low confidence ({confidence:.3f}), expected > 0.5')
"
echo ""

# Test E: Truth-Seeking Question (Bible Verse Test)
run_test "Truth-Seeking Question (Bible Verse Test)" \
    "What does the Bible say about caring for the poor?" \
    "E"

echo "Expected Behaviors for Test E:"
echo "  • Should query Bible verse collection"
echo "  • Should cite Scripture references (book, chapter, verse)"
echo "  • Should include application context for political perspective"
echo "  • Should only use for moral/spiritual questions, not trivial facts"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_E_response.json') as f:
    data = json.load(f)
answer = data.get('answer', '').lower()
keywords = ['verse', 'bible', 'scripture', 'matthew', 'luke', 'john', 'proverbs']
has_scripture = any(keyword in answer for keyword in keywords)
if has_scripture:
    print('   ✓ Scripture references found')
else:
    print('   ✗ No clear Scripture references')
"
echo ""

# Test F: Inverse MarketGuru Triggering (Low Confidence → More Guidance)
run_test "Inverse MarketGuru Triggering (Low Confidence)" \
    "What are Brandon's thoughts on quantum computing policy?" \
    "F"

echo "Expected Behaviors for Test F:"
echo "  • Should have low content confidence (no specific info on quantum computing)"
echo "  • MarketGuru should provide 2x guidance (10+ snippets) to help frame response"
echo "  • Response should acknowledge lack of specific policy but use good communication"
echo "  • Should offer callback for detailed discussion"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_F_response.json') as f:
    data = json.load(f)
confidence = data.get('confidence', 1.0)
offer_callback = data.get('offer_callback', False)
if confidence < 0.5 and offer_callback:
    print(f'   ✓ Low confidence ({confidence:.3f}) with callback offer')
else:
    print(f'   ✗ Expected low confidence with callback (got confidence={confidence:.3f}, callback={offer_callback})')
"
echo ""

# Test G: Generic Comparison with Web Search
run_test "Generic Comparison with Web Search" \
    "Compare Brandon to the Democratic candidates on healthcare" \
    "G"

echo "Expected Behaviors for Test G:"
echo "  • Should identify 'Democratic' as comparison target"
echo "  • Should fetch Brandon's position from BrandonPlatform"
echo "  • Should use web search to research Democratic positions"
echo "  • Should show dual sources: Brandon + external_web"
echo "  • Should offer callback if opponent info incomplete"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_G_response.json') as f:
    data = json.load(f)
sources = data.get('sources', [])
has_brandon = any(s.get('collection') in ['BrandonPlatform', 'brandonsowers_web'] for s in sources)
has_external = any(s.get('collection') in ['PartyPlatform', 'external_web'] for s in sources)
has_dual = data.get('has_dual_sources', False)
if has_brandon and has_external:
    print(f'   ✓ Dual sources found (has_dual_sources={has_dual})')
else:
    print(f'   ✗ Missing dual sources (Brandon={has_brandon}, External={has_external})')
"
echo ""

# Test H: Partial Information Handling
run_test "Partial Information Handling" \
    "How does Brandon's stance on AI regulation compare to Republicans?" \
    "H"

echo "Expected Behaviors for Test H:"
echo "  • Should find some Brandon information (if available)"
echo "  • May lack complete Republican platform info on AI"
echo "  • Response should explain what WAS found and what COULDN'T be determined"
echo "  • Should say 'I found X but couldn't determine Y'"
echo "  • Should always offer callback for incomplete comparisons"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_H_response.json') as f:
    data = json.load(f)
response = data.get('response', '').lower()
offer_callback = data.get('offer_callback', False)
partial_indicators = ['don\\'t have enough', 'couldn\\'t determine', 'need more', 'missing', 'insufficient']
has_partial_language = any(indicator in response for indicator in partial_indicators)
if has_partial_language or offer_callback:
    print(f'   ✓ Partial info handling detected (callback={offer_callback})')
else:
    print(f'   ⚠ May have complete info or lacking partial handling indicators')
"
echo ""

# Test I: Improved Confidence Scoring
run_test "Improved Confidence Scoring (Border Security)" \
    "What is Brandon's position on border security?" \
    "I"

echo "Expected Behaviors for Test I:"
echo "  • Should retrieve from BrandonPlatform with improved similarity calculation"
echo "  • Confidence should be higher than old cap of ~0.46 (distance/2 formula)"
echo "  • With proper formula: distance 0.5 → similarity 0.75, × 1.0 trust = 0.75 confidence"
echo "  • Should NOT offer callback for high-confidence policy answer"
echo ""
echo "🔍 Validation:"
python3.11 -c "
import json
with open('/tmp/test_I_response.json') as f:
    data = json.load(f)
confidence = data.get('confidence', 0)
offer_callback = data.get('offer_callback', True)
if confidence > 0.5 and not offer_callback:
    print(f'   ✓ High confidence ({confidence:.3f}) without callback')
elif confidence > 0.46:
    print(f'   ⚠ Improved confidence ({confidence:.3f}) but may still offer callback')
else:
    print(f'   ✗ Low confidence ({confidence:.3f}), expected > 0.5')
"
echo ""

echo "================================================================================"
echo "📋 Test Summary"
echo "================================================================================"
echo "All test responses saved to /tmp/test_*_response.json for detailed analysis"
echo ""
echo "Key Features Tested:"
echo "  ✓ Dual-source enforcement for comparisons (Tests A, G, H)"
echo "  ✓ Web search for generic comparisons (Test G)"
echo "  ✓ Partial information handling (Test H)"
echo "  ✓ Inverse MarketGuru triggering (Test F)"
echo "  ✓ Improved confidence scoring (Test I)"
echo "  ✓ Bible verse routing (Test E)"
echo "================================================================================"
