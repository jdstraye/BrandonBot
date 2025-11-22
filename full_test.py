#!/usr/bin/env python3
"""
BrandonBot Conversation Logger
Logs all Q&A interactions to CSV for later evaluation by LLM
"""

import csv
import json
import requests
from datetime import datetime
from pathlib import Path

class ConversationLogger:
    def __init__(self, csv_path="brandonbot_conversations.csv"):
        self.csv_path = csv_path
        self.setup_csv()

    def setup_csv(self):
        """Create CSV with headers if it doesn't exist"""
        if not Path(self.csv_path).exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'question',
                    'response',
                    'confidence',
                    'source_count',
                    'collections_used',
                    'has_bible_ref',
                    'has_dual_sources',
                    'offer_callback',
                    'response_length',
                    'question_category',  # For manual tagging
                    'notes'  # For observations
                ])

    def log_conversation(self, question: str, response_data: dict, category: str = "", notes: str = ""):
        """Log a single Q&A interaction"""

        response_text = response_data.get('response', '')
        sources = response_data.get('sources', [])

        # Extract collection names
        collections = list(set(
            s.get('metadata', {}).get('collection', 'unknown') 
            for s in sources
        ))

        # Check for Bible references
        bible_keywords = ['John', 'Proverbs', 'Ephesians', 'Matthew', 'Genesis', 
                         'Leviticus', 'Romans', 'verse', 'scripture', 'Bible']
        has_bible = any(keyword in response_text for keyword in bible_keywords)

        row = [
            datetime.now().isoformat(),
            question,
            response_text,
            response_data.get('confidence', 0),
            len(sources),
            ', '.join(collections),
            has_bible,
            response_data.get('has_dual_sources', False),
            response_data.get('offer_callback', False),
            len(response_text),
            category,
            notes
        ]

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"✓ Logged conversation to {self.csv_path}")

    def wait_for_server(max_wait=60):
        """Wait for server to be ready"""
        import time
        start = time.time()
        while time.time() - start < max_wait:
            try:
                response = requests.get("http://127.0.0.1:5000/")
                if response.status_code == 200:
                    print("✓ Server is ready!")
                    return True
            except:
                print("Waiting for server...")
                time.sleep(2)
        return False

    def query_and_log(self, question: str, category: str = "", notes: str = ""):
        """Query BrandonBot and log the interaction"""
        print(f"\nQuerying: {question}")

        try:
            response = requests.post(
                "http://127.0.0.1:5000/api/query",
                json={"query": question},
                timeout=120
            )
            response_data = response.json()

            # Display response
            print(f"Response: {response_data.get('response', '')[:150]}...")
            print(f"Confidence: {response_data.get('confidence', 0):.2%}")

            # Log it
            self.log_conversation(question, response_data, category, notes)

            return response_data

        except Exception as e:
            print(f"✗ Error: {e} during {question = }")
            return None


def wait_for_server(base_url="http://127.0.0.1:5000", max_wait=60):
    """Wait for server to be ready before running tests"""
    import time
    print(f"Waiting for server at {base_url}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = requests.get(f"{base_url}/")
            if response.status_code == 200:
                print(f"✓ Server is ready! ({int(time.time() - start)}s)")
                return True
        except requests.exceptions.ConnectionError:
            elapsed = int(time.time() - start)
            print(f"  Waiting... ({elapsed}s)", end='\r')
            time.sleep(2)
        except Exception as e:
            print(f"  Unexpected error: {e}")
            time.sleep(2)
    print(f"\n✗ Server didn't respond after {max_wait}s")
    return False


def run_test_suite():
    """Run comprehensive test suite and log all conversations"""
    logger = ConversationLogger()

    test_questions = [
        # A. Economic and Fiscal Policy (AZ-01 Focus: Wealth Preservation & Growth)
        ("What is your plan to address inflation and the rising cost of groceries, gas, and utilities?", "Economic - Inflation", "Cost of living - critical for fixed-income seniors"),
        ("How will you reduce federal income and capital gains taxes?", "Economic - Taxation", "Key for high-net-worth constituents"),
        ("What is your position on federal spending and deficit reduction?", "Economic - Fiscal", "Fiscal responsibility pillar"),
        ("How will you protect Social Security and Medicare for future generations?", "Economic - Entitlements", "High priority for older demographic"),
        ("What will you do about housing affordability and rising home prices?", "Economic - Housing", "Federal role in interest rates and tax incentives"),
        ("How will you support small businesses and reduce regulations?", "Economic - Growth", "Deregulation for high-tech/finance sectors"),

        # B. Arizona-Specific and Infrastructure Issues
        ("What will you do about border security and illegal immigration?", "Arizona - Border", "Top-tier issue for AZ, drug cartels"),
        ("How will you protect Arizona's water rights and the Colorado River?", "Arizona - Water", "Existential issue - desalination, conservation"),
        ("What's your stance on federal land management in Arizona?", "Arizona - Federal Land", "Energy production, limiting federal overreach"),
        ("What is your energy policy for Arizona - natural gas, oil, and solar?", "Arizona - Energy", "Energy independence and affordability"),
        ("How will you improve federal highway funding for the Northeast Valley?", "Arizona - Infrastructure", "Local traffic and commerce corridors"),

        # C. Government and Political Integrity
        ("What are your thoughts on election integrity and voter ID laws?", "Political Integrity - Elections", "Voter confidence, ballot procedures, audits"),
        ("How will you address the national debt crisis?", "Political Integrity - Debt", "Moral and economic hazard of borrowing"),
        ("What will you do to drain the swamp and fight corruption?", "Political Integrity - Ethics", "Campaign finance, dark money, corruption"),
        ("Will you support term limits for Congress?", "Political Integrity - Reform", "Anti-establishment sentiment"),
        ("How will you address Big Tech censorship and protect free speech?", "Political Integrity - Free Speech", "Social media oversight, content moderation"),

        # D. Social and Domestic Issues (Values and Personal Liberty)
        ("How will you lower healthcare costs and prescription drug prices?", "Social - Healthcare", "Market-based reforms, protecting choice"),
        ("What will you do to protect parental rights in education?", "Social - Education", "Parental control over curriculum, school choice"),
        ("What is your position on the Second Amendment and gun rights?", "Social - 2nd Amendment", "Non-negotiable for primary voters"),
        ("What is your position on abortion and reproductive rights?", "Social - Abortion", "Post-Roe landscape, federal vs state role"),
        ("How will you support law enforcement and address the opioid crisis?", "Social - Crime", "Federal support for police, drug crisis"),

        # E. Foreign Policy and National Security
        ("What's your position on China and economic competition?", "Foreign Policy - China", "Trade, supply chain, chip manufacturing"),
        ("What will you do for veterans and their families?", "Foreign Policy - Veterans", "VA healthcare, homelessness, benefits"),
        ("How do you view America's relationship with Israel?", "Foreign Policy - Israel", "Strong foreign policy, global stability"),
        ("What is your stance on cybersecurity and AI regulation?", "Foreign Policy - Cyber/AI", "Protecting infrastructure, AI governance"),
        ("What is your position on trade policy and tariffs?", "Foreign Policy - Trade", "Trade deals benefiting AZ industries"),

        # Faith/Values Questions (Truth-Seeking)
        ("What does the Bible say about helping the poor?", "Faith - Poverty", "Truth-seeking question, expect Bible verses"),
        ("How does your faith inform your politics?", "Faith - Values", "Values alignment, moral foundation"),
        ("What does Scripture teach about justice and fairness?", "Faith - Justice", "Biblical basis for policy positions"),

        # Comparison Questions (Dual-Source Required)
        ("How do you differ from the Republican party on immigration?", "Comparison - GOP", "Internal positioning, dual sources"),
        ("How does your healthcare plan compare to the Democrats?", "Comparison - Democrats", "Policy contrast, dual sources"),
        ("How do you compare to other candidates in this race?", "Comparison - Opponents", "Web search trigger, competitive positioning"),
        

        # Communication Style Tests (MarketGuru Impact)
        ("Hi Brandon, I'm Jayson.", "Communication - Personalization" "Name recognition, personal touch", "Name recognition"),
        ("Hi Brandon, How are you today?", "Communication - Personalization" "Name recognition, personal touch", "small talk"),
        ("Why should I vote for you?", "Persuasion - Value Prop", "MarketGuru impact, awareness-based messaging"),
        ("What makes you different from typical politicians?", "Persuasion - Differentiation", "Unique value proposition, authenticity"),
        ("I'm tired of empty promises. Why should I trust you?", "Persuasion - Trust", "Empathy, credibility building"),

        # Context and Depth Tests
        ("Can you explain your tax plan in more detail?", "Context - Follow-up", "Maintain context, deeper explanation"),
        ("You mentioned border security - what about legal immigration?", "Context - Nuance", "Nuanced position, clarity"),
        ("Let's go, Brandon!", "Context - Underlying meaning","Nuanced position, current events")

        # Hyper-Local Concerns
        ("What will you do for Scottsdale specifically?", "Local - Scottsdale", "Hyper-local concern, community ties"),
        ("How will you address traffic congestion in the Northeast Valley?", "Local - Traffic", "Specific regional issue"),

        # Off-Script / Niche Questions
        ("What are your thoughts on quantum computing policy?", "Edge Case - Tech", "Low confidence test, offer callback"),
        ("How should we regulate cryptocurrency?", "Edge Case - Crypto", "Niche interest, simplified explanation"),
        ("What's your position on space exploration funding?", "Edge Case - Space", "Handle off-script question"),

        # Polarizing Issues (Trust Building)
        ("Some say you're too conservative. How do you respond?", "Polarizing - Ideology", "Handle criticism, build trust"),
        ("How can you represent Democrats in your district?", "Polarizing - Bipartisan", "Engage diverse audiences"),

        # Action and Engagement
        ("How can I help your campaign?", "Engagement - Volunteer", "Drive action, clear next steps"),
        ("Where can I learn more about your policies?", "Engagement - Information", "Encourage further interaction"),

        # Empathy and Personal Connection Tests
        ("My family is struggling to pay for groceries. What will you do?", "Empathy - Economic Hardship", "Personal struggle, empathy required"),
        ("I'm a veteran who can't get VA appointments. Can you help?", "Empathy - Veterans", "Specific pain point, humanize response"),
        ("My daughter's school is teaching things I disagree with. What's your plan?", "Empathy - Education", "Parent concern, acknowledge perspective"),
        ("I lost my job due to regulations. How will you help people like me?", "Empathy - Employment", "Personal impact, empathy + policy"),
        ("My retirement savings have been destroyed by inflation. What will you do?", "Empathy - Seniors", "Financial anxiety, reassurance needed"),

        # Arizona-Specific Voter Concerns (Hyper-Local)
        ("What about the homeless camps near Tempe Town Lake?", "Local - Tempe Homeless", "Specific local issue, community knowledge"),
        ("How will you help with the I-17 corridor traffic problems?", "Local - I-17 Traffic", "Commuter pain point, infrastructure"),
        ("What's your plan for protecting Fountain Hills' water supply?", "Local - Fountain Hills Water", "Small town concern, local knowledge"),
        ("The growth in Queen Creek is overwhelming infrastructure. What will you do?", "Local - Queen Creek Growth", "Fast-growing area concern"),
        ("How will you address the drought affecting Lake Pleasant?", "Local - Lake Pleasant", "Recreation + water issue"),
        ("What about the air quality in Phoenix during dust storm season?", "Local - Air Quality", "Regional health concern"),
        ("Will you support preserving the Superstition Mountains?", "Local - Superstitions", "Conservation, recreation balance"),

        # Context Maintenance (Multi-Turn Simulation)
        ("You mentioned supporting small businesses. How exactly?", "Context - Follow-up Detail", "Request for specifics after general answer"),
        ("That's interesting about taxes. How does that affect someone making $100k?", "Context - Specific Example", "Apply general to specific case"),
        ("Okay, but what about people who can't afford that?", "Context - Challenge Assumption", "Maintain context through pushback"),

        # Complex Topic Simplification
        ("Can you explain the debt ceiling in simple terms?", "Simplification - Debt Ceiling", "Complex national issue, clear explanation"),
        ("What's the difference between Medicare and Medicaid?", "Simplification - Healthcare Programs", "Common confusion, clarity needed"),
        ("How do tariffs actually work and who pays?", "Simplification - Trade", "Economic concept, practical impact"),
        ("What's your position on the Inflation Reduction Act?", "Simplification - Legislation", "Complex bill, clear stance"),

        # Niche/Technical Interests (Diverse Audiences)
        ("What's your stance on water desalination technology for Arizona?", "Niche - Desalination", "Technical solution, informed discussion"),
        ("How do you view nuclear power as an energy solution?", "Niche - Nuclear Energy", "Specific energy policy, nuance"),
        ("What about semiconductor manufacturing and CHIPS Act funding?", "Niche - Semiconductors", "AZ tech industry, specific knowledge"),
        ("Should we reform HOA regulations in Arizona?", "Niche - HOA Reform", "Suburban voter issue, state vs federal"),

        # Trust Building Through Vulnerability
        ("What's something you've changed your mind about?", "Trust - Growth", "Humanize, show capacity to learn"),
        ("What's your biggest weakness as a candidate?", "Trust - Honesty", "Vulnerability, self-awareness"),
        ("Tell me about a time you failed. What did you learn?", "Trust - Failure", "Personal story, relatability"),
        ("Why did you decide to run for Congress?", "Trust - Motivation", "Personal story, authenticity"),

        # Handling Skepticism and Criticism
        ("You're a businessman. How are you different from career politicians?", "Skepticism - Outsider Status", "Turn perceived weakness to strength"),
        ("I've been burned by Republicans before. Why should I trust you?", "Skepticism - Party Loyalty", "Address distrust, build credibility"),
        ("You sound too good to be true. What's the catch?", "Skepticism - Cynicism", "Handle cynicism with empathy"),
        ("How do I know you won't just vote the party line?", "Skepticism - Independence", "Demonstrate independent thinking"),
        ("Politicians always promise things. How are you different?", "Skepticism - Promises", "Credibility, specific commitments"),

        # Diverse Constituency Representation
        ("I'm a Democrat but considering voting for you. Why should I?", "Diverse - Democrat Outreach", "Cross-party appeal, avoid alienation"),
        ("I'm Hispanic and worried about your immigration stance. Explain.", "Diverse - Hispanic Voters", "Sensitive issue, inclusive message"),
        ("I'm a small business owner drowning in regulations. Help?", "Diverse - Business Owner", "Specific constituency, targeted policy"),
        ("I'm a teacher. How will you support public education?", "Diverse - Educators", "Balance parental rights with teacher respect"),
        ("I'm retired military. What have you done for veterans?", "Diverse - Military", "Credibility, specific actions not just words"),

        # Polarizing Issues with Nuance
        ("I'm pro-choice. Can I still vote for you?", "Polarizing - Abortion Nuance", "Avoid alienation, show respect"),
        ("Some Republicans say you're not conservative enough. Respond.", "Polarizing - Conservative Credentials", "Internal party tension"),
        ("How do you balance business interests with environmental protection?", "Polarizing - Environment vs Economy", "Nuanced position, both/and not either/or"),
        ("I support gun rights but want background checks. Where do you stand?", "Polarizing - Gun Rights Nuance", "Find common ground, avoid extremes"),
        ("Should we cut entitlements to fix the debt?", "Polarizing - Entitlement Reform", "Sensitive issue, senior voters, nuance"),

        # Unique Value Proposition
        ("What can you do that the incumbent can't?", "Value Prop - vs Incumbent", "Contrast without attacking"),
        ("Why are you the best choice in the primary?", "Value Prop - Primary", "Differentiation from other Republicans"),
        ("Give me three reasons to vote for you in 30 seconds.", "Value Prop - Elevator Pitch", "Concise, compelling, memorable"),
        ("What's the one issue you care most about?", "Value Prop - Priority", "Clarity of focus, passion"),

        # Testing Articulation of Nuanced Positions
        ("Where do you stand on the federal government's role in healthcare?", "Nuance - Healthcare Role", "Complex position, federal vs market"),
        ("How do you balance immigration enforcement with compassion?", "Nuance - Immigration Balance", "Both/and thinking, humanity + law"),
        ("What's your view on foreign aid while we have problems at home?", "Nuance - Foreign Aid", "Common tension, strategic thinking"),
        ("Should states or the federal government decide abortion laws?", "Nuance - Federalism", "Constitutional nuance, respect states"),

        # Real Voter Frustrations (Arizona-Specific)
        ("Property taxes keep going up. What will you do?", "Frustration - Property Tax", "State vs federal confusion, empathy"),
        ("I'm tired of California transplants changing Arizona. Your thoughts?", "Frustration - Migration", "Cultural concern, avoid alienation"),
        ("Homeless people are everywhere in Phoenix. What's your solution?", "Frustration - Homelessness", "Visible problem, federal role clarity"),
        ("Gas is still too expensive. When will it come down?", "Frustration - Gas Prices", "Immediate concern, realistic timeline"),

        # Engagement and Action (Varied CTAs)
        ("I want to meet you in person. When are you in my area?", "Engagement - Events", "Personal connection, drive attendance"),
        ("Can you come speak at our church/civic group?", "Engagement - Speaking", "Community engagement, accessibility"),
        ("I disagree with you on [issue]. Can we talk about it?", "Engagement - Dialogue", "Openness to disagreement, respect"),
        ("I'm not registered to vote. Should I?", "Engagement - Registration", "Voter registration drive, importance"),
        ("What's the single most important thing I can do to help?", "Engagement - Priority Action", "Focus volunteer energy, clear ask"),
    ]

    print("="*80)
    print("BrandonBot Test Suite - Logging all conversations")
    print("="*80)

    if not wait_for_server():
        print("✗ Server didn't start in time!")
        return

    for question, category, notes in test_questions:
        logger.query_and_log(question, category, notes)
        print()

    print("="*80)
    print(f"All conversations logged to: {logger.csv_path}")
    print("="*80)
    print("\nNext steps:")
    print("1. Open the CSV file")
    print("2. Upload to Claude.ai or ChatGPT")
    print("3. Use the evaluation prompt (see evaluation_prompt.txt)")


def create_evaluation_prompt_file():
    """Create a prompt file for LLM evaluation"""

    prompt = """Evaluate these BrandonBot conversations based on specific criteria.

CONTEXT:
BrandonBot is a chatbot representing Brandon Sowers, a congressional candidate for Arizona's 1st District (AZ-01). The district is conservative, affluent, with many seniors and veterans.

EVALUATION CRITERIA (Score each 1-5):

1. **Clarity**: Is the response easy to understand?
2. **Empathy**: Does it acknowledge the user's perspective?
3. **Accuracy**: Are the facts and policies correct?
4. **Engagement**: Does it encourage further interaction or action?
5. **Tone**: Is it professional yet approachable?
6. **AZ-01 Alignment**: Does it address district-specific concerns?

DISTRICT PRIORITIES:
- Economic: Inflation, taxes, debt reduction, Social Security
- Arizona: Border security, water rights, federal land
- Values: Second Amendment, parental rights, election integrity
- Veterans: VA healthcare, benefits
- Faith: Bible-based moral reasoning for truth-seeking questions

SPECIFIC REQUIREMENTS:
- Policy questions should be specific, not vague corporate-speak
- Comparison questions should cite BOTH Brandon and the other party/candidate
- Faith questions should include actual Bible verses with citations
- Low-confidence answers should offer callback for more info
- Use first-person ("I believe..."), not third-person

INSTRUCTIONS:
For each conversation in the CSV:
1. Score it on all 6 criteria (1-5)
2. Calculate average score
3. Identify strengths
4. Identify areas for improvement
5. Note any red flags (jargon, prompt leakage, factual errors, alienating rhetoric)

OUTPUT FORMAT:
Create a summary table with:
- Question category
- Average scores for each criterion
- Overall grade (A-F)
- Top 3 strengths across all responses
- Top 3 improvement areas
- Specific examples of best and worst responses

Then provide detailed feedback on 3-5 representative conversations.
"""

    with open('evaluation_prompt.txt', 'w') as f:
        f.write(prompt)

    print("✓ Created evaluation_prompt.txt")


if __name__ == "__main__":
    # Create the evaluation prompt file
    create_evaluation_prompt_file()

    # Run test suite
    run_test_suite()

    print("\n" + "="*80)
    print("EVALUATION INSTRUCTIONS")
    print("="*80)
    print("1. Open brandonbot_conversations.csv")
    print("2. Go to an LLM interface (e.g., Claude.ai or ChatGPT")
    print("3. Upload the CSV file")
    print("4. Copy/paste the contents of evaluation_prompt.txt")
    print("="*80)