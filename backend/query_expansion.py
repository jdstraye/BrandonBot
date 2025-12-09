"""
Query Expansion Module for BrandonBot

Static synonym lookup table for expanding user queries with related terms.
This improves retrieval by matching semantic variations without NLP overhead.

Update this manually as campaign messaging evolves.
"""
from typing import List, Set

SYNONYM_TABLE = {
    "healthcare": ["health care", "medical", "medicine", "hospital", "doctor", "insurance", "obamacare", "aca", "affordable care act", "medicare", "medicaid", "prescription", "drugs", "pharmaceutical"],
    "economy": ["economic", "jobs", "employment", "unemployment", "wages", "salary", "income", "business", "commerce", "trade", "gdp", "growth", "recession"],
    "taxes": ["tax", "taxation", "irs", "income tax", "property tax", "sales tax", "tax cuts", "tax reform", "tariffs", "revenue"],
    "immigration": ["immigrant", "immigrants", "border", "borders", "migrant", "migrants", "refugee", "refugees", "visa", "citizenship", "deportation", "illegal", "undocumented", "asylum"],
    "education": ["school", "schools", "college", "university", "student", "students", "teacher", "teachers", "learning", "curriculum", "tuition", "degree"],
    "environment": ["climate", "climate change", "global warming", "pollution", "emissions", "carbon", "green", "renewable", "energy", "solar", "wind", "fossil fuel", "conservation"],
    "guns": ["gun", "firearms", "second amendment", "2nd amendment", "weapons", "nra", "gun control", "gun rights", "ammunition", "rifle", "pistol"],
    "abortion": ["pro-life", "pro-choice", "reproductive", "roe", "roe v wade", "planned parenthood", "pregnancy", "unborn", "fetus"],
    "veterans": ["veteran", "military", "armed forces", "army", "navy", "marines", "air force", "service members", "va", "veterans affairs"],
    "crime": ["criminal", "criminals", "law enforcement", "police", "safety", "security", "prison", "incarceration", "justice", "prosecution"],
    "housing": ["home", "homes", "rent", "rental", "mortgage", "affordable housing", "homelessness", "homeless", "real estate", "property"],
    "infrastructure": ["roads", "bridges", "highways", "transportation", "transit", "public works", "construction", "utilities", "broadband", "internet"],
    "trade": ["tariff", "tariffs", "import", "imports", "export", "exports", "nafta", "usmca", "china trade", "trade deal", "trade war"],
    "social security": ["retirement", "pension", "seniors", "elderly", "aging", "benefits", "entitlements"],
    "foreign policy": ["foreign", "international", "diplomacy", "allies", "nato", "un", "united nations", "war", "peace", "treaty"],
    
    "donate": ["donation", "donations", "contribute", "contribution", "give", "giving", "support", "fund", "fundraise", "money"],
    "volunteer": ["volunteering", "help", "helping", "campaign", "canvass", "phone bank", "door knock", "grassroots"],
    "vote": ["voting", "election", "ballot", "poll", "polls", "primary", "caucus", "registration", "register"],
    "rally": ["event", "events", "town hall", "meeting", "speech", "campaign event", "gathering"],
    
    "brandon": ["brandon sowers", "sowers", "candidate", "our candidate"],
    "republican": ["gop", "conservative", "right", "right-wing"],
    "democrat": ["democratic", "liberal", "left", "left-wing", "dnc"],
    "independent": ["third party", "unaffiliated", "moderate", "centrist"],
    
    "middlemen": ["middle men", "intermediaries", "brokers", "pbm", "pharmacy benefit manager", "insurance company"],
    "farmers": ["farmer", "agriculture", "agricultural", "farm", "farming", "rancher", "ranch", "crop", "crops"],
    "water": ["water rights", "irrigation", "drought", "colorado river", "aquifer", "groundwater"],
    "property rights": ["land rights", "property ownership", "eminent domain", "zoning"],
    "local control": ["states rights", "state rights", "federalism", "decentralization", "local government"],
    
    "faith": ["religion", "religious", "christian", "church", "god", "spiritual", "moral", "morality", "values", "bible", "scripture"],
    "family": ["families", "children", "parents", "marriage", "traditional values"],
    
    "callback": ["call me", "give me a call", "call me back", "phone call", "can we talk", "talk to someone", "speak to someone", "speak with someone", "have someone call", "reach out to me", "schedule a call", "set up a call", "contact me", "get back to me"],
}

TOPIC_KEYWORDS = {
    "comparison": ["compare", "compared", "comparing", "versus", "vs", "difference", "different", "similar", "both", "other candidates", "opponent", "opposition"],
    "statistics": ["percent", "percentage", "number", "numbers", "how many", "how much", "statistics", "data", "rate", "rates"],
    "recent_event": ["recently", "just", "today", "yesterday", "this week", "last week", "news", "latest", "current", "new policy", "announced"],
    "truth_seeking": ["really", "actually", "truth", "true", "honest", "fact", "facts", "verify", "prove", "evidence"],
    "policy": ["position", "stance", "policy", "plan", "proposal", "platform", "believe", "support", "oppose", "vote"],
    "emotional": ["feel", "feeling", "hope", "afraid", "worried", "concerned", "excited", "trust", "believe in", "care about"],
    "callback": ["call", "callback", "phone", "contact", "reach", "talk to", "speak with", "meet"],
}


def expand_query(query: str) -> List[str]:
    """
    Expand a query with synonyms and related terms.
    
    Args:
        query: Original user query
        
    Returns:
        List of expanded query terms (original + synonyms)
    """
    query_lower = query.lower()
    expanded_terms: Set[str] = set()
    
    for key, synonyms in SYNONYM_TABLE.items():
        if key in query_lower:
            expanded_terms.update(synonyms)
        for synonym in synonyms:
            if synonym in query_lower:
                expanded_terms.add(key)
                expanded_terms.update(synonyms)
                break
    
    return list(expanded_terms)


def detect_question_type(query: str) -> List[str]:
    """
    Detect question types based on keywords.
    
    Args:
        query: User query
        
    Returns:
        List of detected question type labels
    """
    query_lower = query.lower()
    detected_types = []
    
    for qtype, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_types.append(qtype)
                break
    
    if not detected_types:
        detected_types.append("policy")
    
    return detected_types


def build_expanded_query(query: str, max_terms: int = 10) -> str:
    """
    Build an expanded search query string.
    
    Args:
        query: Original user query
        max_terms: Maximum number of expansion terms to add
        
    Returns:
        Expanded query string for vector search
    """
    expanded = expand_query(query)
    
    if not expanded:
        return query
    
    top_terms = expanded[:max_terms]
    
    return f"{query} {' '.join(top_terms)}"


def get_topic_from_query(query: str) -> str:
    """
    Identify the primary topic from a query.
    
    Args:
        query: User query
        
    Returns:
        Primary topic string
    """
    query_lower = query.lower()
    
    # Priority phrase detection - check multi-word callback phrases FIRST
    # These take precedence over single-word keyword matching
    callback_phrases = [
        "give me a call", "call me back", "call me", "phone call",
        "can we talk", "talk to someone", "speak to someone", "speak with someone",
        "have someone call", "reach out to me", "schedule a call", "set up a call",
        "contact me", "get back to me", "get in touch", "someone can call"
    ]
    for phrase in callback_phrases:
        if phrase in query_lower:
            return "callback"
    
    topic_scores = {}
    for topic, synonyms in SYNONYM_TABLE.items():
        score = 0
        if topic in query_lower:
            score += 2
        for syn in synonyms:
            if syn in query_lower:
                score += 1
        if score > 0:
            topic_scores[topic] = score
    
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    
    return "general"
