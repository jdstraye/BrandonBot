#!/usr/bin/env python3
"""
Seed Weaviate with test data for validation.

This script loads:
1. FECProhibited phrases for compliance checking
2. BrandonPlatform content for policy questions (including abortion)
3. Sample PreviousQA data

Run with: python -m seed_weaviate
"""

import asyncio
import logging
from weaviate_manager import WeaviateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BRANDON_PLATFORM_DATA = [
    {
        "content": "Brandon believes in protecting the sanctity of life. His stance on abortion is grounded in his faith and values. He supports reasonable restrictions on late-term abortion while respecting the complexities families face. Brandon advocates for increased support for adoption services, crisis pregnancy centers, and maternal healthcare to provide women with real choices and support.",
        "source": "Brandon Campaign Platform - Life Issues",
        "category": "social_policy",
    },
    {
        "content": "On reproductive rights and abortion policy, Brandon takes a principled pro-life position. He believes life begins at conception and deserves protection. However, he also recognizes the need for compassionate approaches that support mothers and families. His policy focuses on reducing abortion through positive alternatives rather than purely punitive measures.",
        "source": "Brandon Town Hall Q&A - March 2024",
        "category": "social_policy",
    },
    {
        "content": "Brandon supports lower taxes for working families and small businesses. He believes government spending should be fiscally responsible and that we should reduce the national debt for future generations.",
        "source": "Brandon Campaign Platform - Economic Policy",
        "category": "economic_policy",
    },
    {
        "content": "Brandon is committed to securing our borders while maintaining America's tradition as a nation of legal immigrants. He supports comprehensive immigration reform that is fair, humane, and enforces existing laws.",
        "source": "Brandon Campaign Platform - Immigration",
        "category": "immigration",
    },
    {
        "content": "Brandon believes in the Second Amendment right to bear arms while supporting common-sense measures to keep guns out of the hands of criminals and the mentally ill. He opposes gun bans but supports background check improvements.",
        "source": "Brandon Campaign Platform - Second Amendment",
        "category": "constitutional_rights",
    },
    {
        "content": "Brandon supports school choice and parental rights in education. He believes parents should have the freedom to choose the best educational options for their children, whether public, private, charter, or homeschool.",
        "source": "Brandon Campaign Platform - Education",
        "category": "education",
    },
]

PREVIOUS_QA_DATA = [
    {
        "content": "Q: What is Brandon's position on healthcare? A: Brandon supports market-based healthcare solutions that increase competition and lower costs. He believes in protecting coverage for pre-existing conditions while giving families more choices. He opposes government-run single-payer healthcare.",
        "source": "Previous Q&A Database",
        "category": "healthcare",
    },
    {
        "content": "Q: Where does Brandon stand on climate and energy policy? A: Brandon supports an all-of-the-above energy approach that includes traditional and renewable sources. He believes in energy independence and opposes policies that would dramatically raise energy costs for working families.",
        "source": "Previous Q&A Database",
        "category": "energy",
    },
]

FEC_PROHIBITED_DATA = [
    {
        "content": "PROHIBITED: Making tax deductibility claims for political contributions. Phrases like 'your donation is tax deductible' or 'tax write-off for contributions' are strictly prohibited as political contributions are NOT tax deductible. Violation: 11 CFR 110.11",
        "source": "FEC Regulations - 11 CFR 110.11",
        "category": "tax_advice",
    },
    {
        "content": "PROHIBITED: Soliciting or processing financial transactions. The bot must never request credit card numbers, bank account information, or process donations directly. All donations must go through the official, FEC-compliant donation portal. Phrases like 'enter your credit card' or 'provide payment information' are prohibited.",
        "source": "FEC Regulations - Financial Solicitation",
        "category": "financial_solicitation",
    },
    {
        "content": "PROHIBITED: Making defamatory statements about opponents. Statements like 'is a criminal', 'committed fraud', 'stole money', 'is corrupt', or 'took bribes' without verified factual basis constitute defamation. Focus on policy differences, not personal attacks.",
        "source": "FEC Regulations - Defamation Guidelines",
        "category": "defamation",
    },
    {
        "content": "PROHIBITED: Claiming to be the candidate or a human. The AI assistant must never claim 'I am Brandon', 'I am the candidate', 'I am a human', or 'speaking as the candidate'. It must always identify as an AI assistant for the campaign.",
        "source": "FEC Regulations - Identity Disclosure",
        "category": "false_identity",
    },
    {
        "content": "PROHIBITED: Making unverified endorsement claims or guarantees. Statements like 'endorsed by [organization]' without verification, 'guaranteed to win', 'will definitely happen', or '100% certain' are prohibited. Campaign promises must be aspirational, not absolute.",
        "source": "FEC Regulations - False Claims",
        "category": "false_claims",
    },
    {
        "content": "PROHIBITED: Coercive language regarding support or donations. Phrases like 'vote for us or else', 'if you don't support us', 'you must donate', or 'failure to support' are prohibited. All supporter engagement must be voluntary and respectful.",
        "source": "FEC Regulations - Anti-Coercion",
        "category": "coercion",
    },
    {
        "content": "PROHIBITED: Providing medical advice or treatment recommendations. The campaign bot must never suggest medications, treatments, or make health claims. Phrases like 'you should take', 'this treatment will cure', or 'I recommend this medication' are strictly prohibited.",
        "source": "FEC Regulations - Medical Advice",
        "category": "medical_advice",
    },
]


async def seed_collection(weaviate: WeaviateManager, collection_name: str, data: list):
    """Seed a collection with data, skipping if already populated."""
    count = await weaviate.get_collection_count(collection_name)
    if count > 0:
        logger.info(f"{collection_name} already has {count} documents, skipping seed")
        return count
    
    added = 0
    for item in data:
        success = await weaviate.add_document(
            collection_name=collection_name,
            content=item["content"],
            source=item["source"],
            category=item.get("category", ""),
        )
        if success:
            added += 1
    
    logger.info(f"Added {added} documents to {collection_name}")
    return added


async def main():
    logger.info("Starting Weaviate seed process...")
    
    weaviate = WeaviateManager()
    await weaviate.initialize()
    
    logger.info("Seeding BrandonPlatform...")
    await seed_collection(weaviate, "BrandonPlatform", BRANDON_PLATFORM_DATA)
    
    logger.info("Seeding PreviousQA...")
    await seed_collection(weaviate, "PreviousQA", PREVIOUS_QA_DATA)
    
    logger.info("Seeding FECProhibited...")
    await seed_collection(weaviate, "FECProhibited", FEC_PROHIBITED_DATA)
    
    logger.info("Verifying collection counts...")
    for collection in ["BrandonPlatform", "PreviousQA", "PartyPlatform", "MarketGurus", "FECProhibited"]:
        count = await weaviate.get_collection_count(collection)
        logger.info(f"  {collection}: {count} documents")
    
    await weaviate.close()
    logger.info("Seed complete!")


if __name__ == "__main__":
    asyncio.run(main())
