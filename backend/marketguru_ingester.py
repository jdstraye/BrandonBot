#!/usr/bin/env python3
"""
MarketGuru Smart Ingestion System for BrandonBot
Chunks copywriting wisdom by Rule/Example (not paragraphs)
Adds rich metadata: tone, CTA type, question type (Ogilvy/Schwartz), topic
"""
import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Question types based on Schwartz awareness stages and Ogilvy categories"""
    UNAWARE = "unaware"
    PROBLEM_AWARE = "problem_aware"
    SOLUTION_AWARE = "solution_aware"
    PRODUCT_AWARE = "product_aware"
    MOST_AWARE = "most_aware"
    OPPOSITIONAL = "oppositional"
    SKEPTICAL = "skeptical"
    SEEKING_PROOF = "seeking_proof"
    COMPARISON = "comparison"
    SIMPLE_INQUIRY = "simple_inquiry"
    TRUST_BUILDING = "trust_building"
    EMOTIONAL_APPEAL = "emotional_appeal"


class Tone(str, Enum):
    """Response tone categories"""
    ASPIRATIONAL = "aspirational"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"
    URGENT = "urgent"
    REASSURING = "reassuring"
    EDUCATIONAL = "educational"
    PERSUASIVE = "persuasive"
    STORYTELLING = "storytelling"
    DIRECT = "direct"


class CTAType(str, Enum):
    """Call-to-action types"""
    VOLUNTEER = "volunteer"
    DONATE = "donate"
    SHARE = "share"
    LEARN_MORE = "learn_more"
    CONTACT = "contact"
    CALLBACK = "callback"
    ATTEND_EVENT = "attend_event"
    SIGN_PETITION = "sign_petition"
    NONE = "none"


class Topic(str, Enum):
    """Political topic categories"""
    ECONOMY = "economy"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    IMMIGRATION = "immigration"
    ENVIRONMENT = "environment"
    FOREIGN_POLICY = "foreign_policy"
    TAXES = "taxes"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    GENERAL = "general"
    VALUES = "values"
    LEADERSHIP = "leadership"


@dataclass
class MarketGuruChunk:
    """A single rule/example chunk with rich metadata"""
    content: str
    source: str
    guru_name: str
    chunk_type: str
    question_types: List[str]
    tones: List[str]
    cta_types: List[str]
    topics: List[str]
    framework: str
    rule_number: Optional[int] = None
    example_text: Optional[str] = None
    anti_pattern: Optional[str] = None
    
    def to_metadata_dict(self) -> Dict:
        return {
            "guru_name": self.guru_name,
            "chunk_type": self.chunk_type,
            "question_types": json.dumps(self.question_types),
            "tones": json.dumps(self.tones),
            "cta_types": json.dumps(self.cta_types),
            "topics": json.dumps(self.topics),
            "framework": self.framework,
            "rule_number": self.rule_number,
            "has_example": bool(self.example_text),
            "has_anti_pattern": bool(self.anti_pattern)
        }


class MarketGuruParser:
    """Parses marketing wisdom documents into structured chunks"""
    
    @staticmethod
    def parse_schwartz_awareness(text: str, source: str) -> List[MarketGuruChunk]:
        """Parse Eugene Schwartz's 5 Stages of Awareness document"""
        chunks = []
        
        stage_patterns = {
            "STAGE 1: UNAWARE": QuestionType.UNAWARE,
            "STAGE 2: PROBLEM AWARE": QuestionType.PROBLEM_AWARE,
            "STAGE 3: SOLUTION AWARE": QuestionType.SOLUTION_AWARE,
            "STAGE 4: PRODUCT AWARE": QuestionType.PRODUCT_AWARE,
            "STAGE 5: MOST AWARE": QuestionType.MOST_AWARE,
        }
        
        stage_sections = re.split(r'###\s+(STAGE \d+:[^\n]+)', text)
        
        for i in range(1, len(stage_sections), 2):
            stage_title = stage_sections[i].strip()
            stage_content = stage_sections[i + 1] if i + 1 < len(stage_sections) else ""
            
            question_type = None
            for pattern, qtype in stage_patterns.items():
                if pattern in stage_title:
                    question_type = qtype
                    break
            
            if not question_type:
                continue
            
            customer_state = re.search(r'\*\*Customer State:\*\*\s*\n([^\n]+)', stage_content)
            thinking = re.search(r'\*\*What They\'re Thinking:\*\*\s*\n([^\n]+)', stage_content)
            goal = re.search(r'\*\*Your Marketing Goal:\*\*\s*\n([^\n]+)', stage_content)
            strategy_match = re.search(r'\*\*Message Strategy:\*\*\s*\n([\s\S]*?)(?=\*\*Copy Style:|$)', stage_content)
            
            examples = re.findall(r'(?:GOOD|Example):\s*"([^"]+)"', stage_content)
            anti_patterns = re.findall(r'BAD:\s*"([^"]+)"', stage_content)
            
            tone = Tone.EDUCATIONAL
            if question_type == QuestionType.UNAWARE:
                tone = Tone.STORYTELLING
            elif question_type == QuestionType.PROBLEM_AWARE:
                tone = Tone.EMPATHETIC
            elif question_type == QuestionType.SOLUTION_AWARE:
                tone = Tone.AUTHORITATIVE
            elif question_type == QuestionType.PRODUCT_AWARE:
                tone = Tone.PERSUASIVE
            elif question_type == QuestionType.MOST_AWARE:
                tone = Tone.URGENT
            
            rule_content = f"""SCHWARTZ AWARENESS STAGE: {stage_title}

CUSTOMER STATE: {customer_state.group(1) if customer_state else 'Unknown'}

WHAT THEY'RE THINKING: {thinking.group(1) if thinking else 'Unknown'}

YOUR MARKETING GOAL: {goal.group(1) if goal else 'Unknown'}

MESSAGE STRATEGY:
{strategy_match.group(1).strip() if strategy_match else 'See full document'}

EXAMPLE OF GOOD COPY:
{examples[0] if examples else 'None provided'}
"""
            
            chunk = MarketGuruChunk(
                content=rule_content,
                source=source,
                guru_name="Eugene Schwartz",
                chunk_type="awareness_stage",
                question_types=[question_type.value],
                tones=[tone.value],
                cta_types=[CTAType.LEARN_MORE.value if question_type in [QuestionType.UNAWARE, QuestionType.PROBLEM_AWARE] 
                          else CTAType.DONATE.value],
                topics=[Topic.GENERAL.value],
                framework="5_stages_of_awareness",
                rule_number=list(stage_patterns.keys()).index(stage_title.split('(')[0].strip()) + 1 if stage_title else None,
                example_text=examples[0] if examples else None,
                anti_pattern=anti_patterns[0] if anti_patterns else None
            )
            chunks.append(chunk)
        
        cardinal_rule = re.search(r'## THE CARDINAL RULE\s*([\s\S]*?)(?=##|$)', text)
        if cardinal_rule:
            chunks.append(MarketGuruChunk(
                content=f"""SCHWARTZ CARDINAL RULE:
{cardinal_rule.group(1).strip()}

NEVER skip stages in your messaging. Match your message to where the prospect IS, not where you want them to be.""",
                source=source,
                guru_name="Eugene Schwartz",
                chunk_type="cardinal_rule",
                question_types=[qt.value for qt in QuestionType],
                tones=[Tone.AUTHORITATIVE.value],
                cta_types=[CTAType.NONE.value],
                topics=[Topic.GENERAL.value],
                framework="5_stages_of_awareness",
                rule_number=0
            ))
        
        return chunks

    @staticmethod
    def parse_ogilvy_commandments(text: str, source: str) -> List[MarketGuruChunk]:
        """Parse David Ogilvy's 10 Commandments document"""
        chunks = []
        
        commandment_pattern = r'###\s*(\d+)\.\s+([A-Z][A-Z\s\']+)\n\n([^#]+?)(?=###|\Z)'
        matches = re.findall(commandment_pattern, text, re.DOTALL)
        
        for num, title, content in matches:
            quotes = re.findall(r'"([^"]+)"', content)
            key_points = re.findall(r'-\s*([^\n]+)', content)
            
            question_types = [QuestionType.SIMPLE_INQUIRY.value]
            tones = [Tone.AUTHORITATIVE.value]
            
            title_lower = title.lower()
            if "bore" in title_lower:
                question_types.append(QuestionType.UNAWARE.value)
                tones.append(Tone.STORYTELLING.value)
            if "fact" in title_lower:
                question_types.append(QuestionType.SEEKING_PROOF.value)
                tones.append(Tone.EDUCATIONAL.value)
            if "sell" in title_lower:
                question_types.append(QuestionType.PRODUCT_AWARE.value)
                tones.append(Tone.PERSUASIVE.value)
            if "idea" in title_lower:
                question_types.append(QuestionType.SOLUTION_AWARE.value)
            if "brand" in title_lower or "personality" in title_lower:
                question_types.append(QuestionType.TRUST_BUILDING.value)
            
            rule_content = f"""OGILVY COMMANDMENT #{num}: {title.strip()}

CORE PRINCIPLE:
{quotes[0] if quotes else content[:200]}

KEY POINTS:
{chr(10).join('• ' + kp for kp in key_points[:5])}

APPLICATION:
{content[content.find('Application:'):content.find('Application:') + 300] if 'Application:' in content else 'Apply this principle to all campaign messaging.'}
"""
            
            chunk = MarketGuruChunk(
                content=rule_content,
                source=source,
                guru_name="David Ogilvy",
                chunk_type="commandment",
                question_types=list(set(question_types)),
                tones=list(set(tones)),
                cta_types=[CTAType.NONE.value],
                topics=[Topic.GENERAL.value, Topic.LEADERSHIP.value],
                framework="10_commandments",
                rule_number=int(num),
                example_text=quotes[1] if len(quotes) > 1 else None
            )
            chunks.append(chunk)
        
        headline_section = re.search(r"## OGILVY'S HEADLINE PRINCIPLES([\s\S]*?)(?=##|$)", text)
        if headline_section:
            chunks.append(MarketGuruChunk(
                content=f"""OGILVY HEADLINE PRINCIPLES:
{headline_section.group(1).strip()}

KEY INSIGHT: 80% of readers never get past the headline. Make it count.""",
                source=source,
                guru_name="David Ogilvy",
                chunk_type="headline_principles",
                question_types=[QuestionType.UNAWARE.value, QuestionType.PROBLEM_AWARE.value],
                tones=[Tone.DIRECT.value, Tone.PERSUASIVE.value],
                cta_types=[CTAType.LEARN_MORE.value],
                topics=[Topic.GENERAL.value],
                framework="headline_formula"
            ))
        
        return chunks

    @staticmethod
    def parse_collier_principles(text: str, source: str) -> List[MarketGuruChunk]:
        """Parse Robert Collier's Letter Book principles"""
        chunks = []
        
        principle_pattern = r'###\s*(\d+)\.\s+([A-Z][A-Z\s]+)\n([^#]+?)(?=###|\Z)'
        matches = re.findall(principle_pattern, text, re.DOTALL)
        
        for num, title, content in matches:
            quotes = re.findall(r'"([^"]+)"', content)
            key_points = re.findall(r'-\s*([^\n]+)', content)
            examples = re.findall(r'Example:\s*([^\n]+)', content)
            
            question_types = [QuestionType.SIMPLE_INQUIRY.value]
            tones = [Tone.PERSUASIVE.value]
            
            title_lower = title.lower()
            if "desire" in title_lower or "visualized" in title_lower:
                question_types.extend([QuestionType.EMOTIONAL_APPEAL.value, QuestionType.UNAWARE.value])
                tones.append(Tone.ASPIRATIONAL.value)
            if "wedge" in title_lower or "entering" in title_lower:
                question_types.append(QuestionType.PROBLEM_AWARE.value)
                tones.append(Tone.EMPATHETIC.value)
            if "narrative" in title_lower or "story" in title_lower:
                tones.append(Tone.STORYTELLING.value)
            if "closing" in title_lower:
                question_types.append(QuestionType.MOST_AWARE.value)
                tones.append(Tone.URGENT.value)
            if "proof" in title_lower or "appeal" in title_lower:
                question_types.append(QuestionType.SEEKING_PROOF.value)
            
            rule_content = f"""COLLIER PRINCIPLE #{num}: {title.strip()}

CORE WISDOM:
{quotes[0] if quotes else 'See below for key points.'}

KEY POINTS:
{chr(10).join('• ' + kp for kp in key_points[:5])}

PRACTICAL EXAMPLE:
{examples[0] if examples else 'Apply this to frame your political message.'}
"""
            
            chunk = MarketGuruChunk(
                content=rule_content,
                source=source,
                guru_name="Robert Collier",
                chunk_type="principle",
                question_types=list(set(question_types)),
                tones=list(set(tones)),
                cta_types=[CTAType.CONTACT.value, CTAType.VOLUNTEER.value],
                topics=[Topic.GENERAL.value, Topic.VALUES.value],
                framework="letter_book_principles",
                rule_number=int(num),
                example_text=examples[0] if examples else None
            )
            chunks.append(chunk)
        
        psychological_section = re.search(r"## PSYCHOLOGICAL TRIGGERS([\s\S]*?)(?=##|$)", text)
        if psychological_section:
            triggers = ["Scarcity", "Social Proof", "Authority", "Reciprocity", "Consistency"]
            for trigger in triggers:
                trigger_match = re.search(rf"###\s*{trigger}\s*\n([^\n]+)\n([^\n]+)?", psychological_section.group(1))
                if trigger_match:
                    chunks.append(MarketGuruChunk(
                        content=f"""COLLIER PSYCHOLOGICAL TRIGGER: {trigger.upper()}

EXAMPLES:
• {trigger_match.group(1)}
• {trigger_match.group(2) if trigger_match.group(2) else ''}

USE THIS WHEN: You need to move a prospect from consideration to action.""",
                        source=source,
                        guru_name="Robert Collier",
                        chunk_type="psychological_trigger",
                        question_types=[QuestionType.PRODUCT_AWARE.value, QuestionType.MOST_AWARE.value],
                        tones=[Tone.URGENT.value, Tone.PERSUASIVE.value],
                        cta_types=[CTAType.DONATE.value, CTAType.VOLUNTEER.value],
                        topics=[Topic.GENERAL.value],
                        framework="psychological_triggers"
                    ))
        
        return chunks

    @staticmethod  
    def parse_generic_marketing_doc(text: str, source: str, guru_name: str) -> List[MarketGuruChunk]:
        """Parse any marketing document into rule-based chunks"""
        chunks = []
        
        sections = re.split(r'\n##\s+', text)
        
        for section in sections[1:]:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            title = lines[0].strip()
            content = '\n'.join(lines[1:]).strip()
            
            if len(content) < 50:
                continue
            
            question_types = [QuestionType.SIMPLE_INQUIRY.value]
            tones = [Tone.EDUCATIONAL.value]
            
            content_lower = content.lower()
            if any(word in content_lower for word in ['story', 'narrative', 'tell']):
                tones.append(Tone.STORYTELLING.value)
            if any(word in content_lower for word in ['urgent', 'now', 'limited', 'deadline']):
                tones.append(Tone.URGENT.value)
                question_types.append(QuestionType.MOST_AWARE.value)
            if any(word in content_lower for word in ['proof', 'evidence', 'data', 'research']):
                question_types.append(QuestionType.SEEKING_PROOF.value)
                tones.append(Tone.AUTHORITATIVE.value)
            if any(word in content_lower for word in ['feel', 'emotion', 'heart', 'care']):
                question_types.append(QuestionType.EMOTIONAL_APPEAL.value)
                tones.append(Tone.EMPATHETIC.value)
            
            chunk = MarketGuruChunk(
                content=f"""{guru_name.upper()} - {title}

{content[:1500]}""",
                source=source,
                guru_name=guru_name,
                chunk_type="section",
                question_types=list(set(question_types)),
                tones=list(set(tones)),
                cta_types=[CTAType.LEARN_MORE.value],
                topics=[Topic.GENERAL.value],
                framework="general_marketing"
            )
            chunks.append(chunk)
        
        return chunks


class MarketGuruIngester:
    """Main ingestion class for MarketGuru documents"""
    
    def __init__(self, weaviate_manager):
        self.weaviate = weaviate_manager
        self.parser = MarketGuruParser()
    
    def read_file(self, file_path: str) -> str:
        """Read text from various file formats"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            import pypdf
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                return '\n'.join(page.extract_text() for page in reader.pages)
        elif ext == '.html':
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n')
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
    
    def identify_document_type(self, text: str, filename: str) -> Tuple[str, str]:
        """Identify the guru and framework from document content"""
        filename_lower = filename.lower()
        text_lower = text[:2000].lower()
        
        if 'schwartz' in filename_lower or '5 stages' in text_lower or 'awareness' in text_lower:
            return "Eugene Schwartz", "5_stages_of_awareness"
        elif 'ogilvy' in filename_lower or 'commandments' in text_lower:
            return "David Ogilvy", "10_commandments"
        elif 'collier' in filename_lower or 'letter book' in text_lower:
            return "Robert Collier", "letter_book_principles"
        elif 'boron' in filename_lower or 'halbert' in text_lower:
            return "Gary Halbert", "boron_letters"
        elif 'hopkins' in filename_lower or 'scientific advertising' in text_lower:
            return "Claude Hopkins", "scientific_advertising"
        elif 'caples' in filename_lower or 'tested advertising' in text_lower:
            return "John Caples", "tested_advertising_methods"
        else:
            return "Unknown Guru", "general_marketing"
    
    def parse_document(self, file_path: str) -> List[MarketGuruChunk]:
        """Parse a document into structured chunks based on its type"""
        text = self.read_file(file_path)
        if not text:
            return []
        
        filename = Path(file_path).name
        guru_name, framework = self.identify_document_type(text, filename)
        
        logger.info(f"  Identified: {guru_name} / {framework}")
        
        if framework == "5_stages_of_awareness":
            chunks = self.parser.parse_schwartz_awareness(text, filename)
        elif framework == "10_commandments":
            chunks = self.parser.parse_ogilvy_commandments(text, filename)
        elif framework == "letter_book_principles":
            chunks = self.parser.parse_collier_principles(text, filename)
        else:
            chunks = self.parser.parse_generic_marketing_doc(text, filename, guru_name)
        
        if not chunks:
            chunks = self.parser.parse_generic_marketing_doc(text, filename, guru_name)
        
        return chunks
    
    async def ingest_chunk(self, chunk: MarketGuruChunk) -> bool:
        """Ingest a single chunk into Weaviate MarketGurus collection"""
        try:
            metadata = chunk.to_metadata_dict()
            
            success = await self.weaviate.add_document(
                collection_name="MarketGurus",
                content=chunk.content,
                source=chunk.source,
                category=chunk.framework,
                metadata=metadata
            )
            return success
        except Exception as e:
            logger.error(f"Failed to ingest chunk: {e}")
            return False
    
    async def ingest_directory(self, directory: str) -> Dict[str, int]:
        """Ingest all marketing documents from a directory"""
        stats = {"files": 0, "chunks": 0, "errors": 0}
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return stats
        
        supported_extensions = {'.txt', '.pdf', '.html'}
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() not in supported_extensions:
                continue
            
            logger.info(f"\n📄 Processing: {file_path.name}")
            stats["files"] += 1
            
            try:
                chunks = self.parse_document(str(file_path))
                logger.info(f"  → Parsed {len(chunks)} chunks")
                
                for chunk in chunks:
                    if await self.ingest_chunk(chunk):
                        stats["chunks"] += 1
                    else:
                        stats["errors"] += 1
                        
            except Exception as e:
                logger.error(f"  ✗ Error processing {file_path.name}: {e}")
                stats["errors"] += 1
        
        return stats


async def main():
    """Run MarketGuru ingestion"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from weaviate_manager import WeaviateManager
    
    logger.info("=" * 70)
    logger.info("MarketGuru Smart Ingestion System")
    logger.info("Chunking by Rule/Example with Rich Metadata")
    logger.info("=" * 70)
    
    weaviate = WeaviateManager("./weaviate_data")
    await weaviate.initialize()
    
    ingester = MarketGuruIngester(weaviate)
    
    market_gurus_dir = "../documents/market_gurus"
    if not os.path.exists(market_gurus_dir):
        market_gurus_dir = "./documents/market_gurus"
    
    stats = await ingester.ingest_directory(market_gurus_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("Ingestion Complete!")
    logger.info(f"  Files processed: {stats['files']}")
    logger.info(f"  Chunks created: {stats['chunks']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info("=" * 70)
    
    count = await weaviate.get_collection_count("MarketGurus")
    logger.info(f"\nMarketGurus collection now has {count} total chunks")
    
    await weaviate.close()


if __name__ == "__main__":
    asyncio.run(main())
