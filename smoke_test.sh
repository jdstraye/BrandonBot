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
    answer=$(echo "$response" | python3.11 -c "import sys, json; data=json.load(sys.stdin); print(data.get('answer', '')[:300])" 2>/dev/null)
    
    echo "✅ Response received"
    echo ""
    echo "📊 Response Analysis:"
    echo "   Confidence: $confidence"
    echo "   Sources: $source_count"
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

echo "================================================================================"
echo "📋 Test Summary"
echo "================================================================================"
echo "All test responses saved to /tmp/test_*_response.json for detailed analysis"
echo "================================================================================"
