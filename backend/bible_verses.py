"""
Bible Verse Collection for BrandonBot
Curated verses for moral/spiritual guidance on political topics
"""

from typing import Dict, List, Optional

class BibleVerseCollection:
    """Structured collection mapping topics to relevant Bible verses"""
    
    VERSES: Dict[str, List[Dict[str, str]]] = {
        "immigration": [
            {
                "reference": "Leviticus 19:34",
                "text": "The foreigner residing among you must be treated as your native-born. Love them as yourself, for you were foreigners in Egypt. I am the Lord your God.",
                "context": "Biblical call to compassion for immigrants and strangers"
            },
            {
                "reference": "Hebrews 13:2",
                "text": "Do not forget to show hospitality to strangers, for by so doing some people have shown hospitality to angels without knowing it.",
                "context": "Hospitality and kindness to strangers as spiritual practice"
            },
            {
                "reference": "Matthew 25:35",
                "text": "For I was hungry and you gave me something to eat, I was thirsty and you gave me something to drink, I was a stranger and you invited me in.",
                "context": "Christ's teaching on welcoming the stranger"
            }
        ],
        "stewardship": [
            {
                "reference": "Genesis 1:28",
                "text": "God blessed them and said to them, 'Be fruitful and increase in number; fill the earth and subdue it. Rule over the fish in the sea and the birds in the sky and over every living creature that moves on the ground.'",
                "context": "Divine mandate for responsible stewardship of creation"
            },
            {
                "reference": "Psalm 24:1",
                "text": "The earth is the Lord's, and everything in it, the world, and all who live in it.",
                "context": "Reminder that creation belongs to God, we are caretakers"
            },
            {
                "reference": "Numbers 35:33",
                "text": "Do not pollute the land where you are. Bloodshed pollutes the land, and atonement cannot be made for the land on which blood has been shed, except by the blood of the one who shed it.",
                "context": "Biblical prohibition against polluting the land"
            }
        ],
        "justice": [
            {
                "reference": "Micah 6:8",
                "text": "He has shown you, O mortal, what is good. And what does the Lord require of you? To act justly and to love mercy and to walk humbly with your God.",
                "context": "Core definition of justice, mercy, and humility"
            },
            {
                "reference": "Proverbs 31:8-9",
                "text": "Speak up for those who cannot speak for themselves, for the rights of all who are destitute. Speak up and judge fairly; defend the rights of the poor and needy.",
                "context": "Advocacy for the vulnerable and voiceless"
            },
            {
                "reference": "Isaiah 1:17",
                "text": "Learn to do right; seek justice. Defend the oppressed. Take up the cause of the fatherless; plead the case of the widow.",
                "context": "Call to active pursuit of justice for marginalized"
            }
        ],
        "truth": [
            {
                "reference": "John 8:32",
                "text": "Then you will know the truth, and the truth will set you free.",
                "context": "Christ's teaching on the liberating power of truth"
            },
            {
                "reference": "Proverbs 12:22",
                "text": "The Lord detests lying lips, but he delights in people who are trustworthy.",
                "context": "God's value for honesty and integrity"
            },
            {
                "reference": "Ephesians 4:25",
                "text": "Therefore each of you must put off falsehood and speak truthfully to your neighbor, for we are all members of one body.",
                "context": "Call to truthfulness in community"
            }
        ],
        "integrity": [
            {
                "reference": "Proverbs 11:3",
                "text": "The integrity of the upright guides them, but the unfaithful are destroyed by their duplicity.",
                "context": "Integrity as moral compass for leadership"
            },
            {
                "reference": "Psalm 15:1-2",
                "text": "Lord, who may dwell in your sacred tent? Who may live on your holy mountain? The one whose walk is blameless, who does what is righteous, who speaks the truth from their heart.",
                "context": "Standards for godly leadership and character"
            },
            {
                "reference": "1 Timothy 3:2-3",
                "text": "Now the overseer is to be above reproach, faithful to his wife, temperate, self-controlled, respectable, hospitable, able to teach, not given to drunkenness, not violent but gentle, not quarrelsome, not a lover of money.",
                "context": "Biblical qualifications for leadership"
            }
        ],
        "compassion": [
            {
                "reference": "Colossians 3:12",
                "text": "Therefore, as God's chosen people, holy and dearly loved, clothe yourselves with compassion, kindness, humility, gentleness and patience.",
                "context": "Call to embody compassion in all interactions"
            },
            {
                "reference": "Proverbs 19:17",
                "text": "Whoever is kind to the poor lends to the Lord, and he will reward them for what they have done.",
                "context": "Kindness to the poor as service to God"
            },
            {
                "reference": "Matthew 9:36",
                "text": "When he saw the crowds, he had compassion on them, because they were harassed and helpless, like sheep without a shepherd.",
                "context": "Christ's model of compassionate leadership"
            }
        ],
        "family": [
            {
                "reference": "Ephesians 6:4",
                "text": "Fathers, do not exasperate your children; instead, bring them up in the training and instruction of the Lord.",
                "context": "Parental responsibility for godly upbringing"
            },
            {
                "reference": "Proverbs 22:6",
                "text": "Start children off on the way they should go, and even when they are old they will not turn from it.",
                "context": "Importance of early moral foundation"
            },
            {
                "reference": "1 Timothy 5:8",
                "text": "Anyone who does not provide for their relatives, and especially for their own household, has denied the faith and is worse than an unbeliever.",
                "context": "Family responsibility and provision"
            }
        ],
        "authority": [
            {
                "reference": "Romans 13:1-2",
                "text": "Let everyone be subject to the governing authorities, for there is no authority except that which God has established. The authorities that exist have been established by God. Consequently, whoever rebels against the authority is rebelling against what God has instituted, and those who do so will bring judgment on themselves.",
                "context": "Biblical view of governmental authority and responsibility"
            },
            {
                "reference": "1 Peter 2:13-14",
                "text": "Submit yourselves for the Lord's sake to every human authority: whether to the emperor, as the supreme authority, or to governors, who are sent by him to punish those who do wrong and to commend those who do right.",
                "context": "Civic duty and respect for authority"
            }
        ],
        "wealth": [
            {
                "reference": "1 Timothy 6:10",
                "text": "For the love of money is a root of all kinds of evil. Some people, eager for money, have wandered from the faith and pierced themselves with many griefs.",
                "context": "Warning against greed and materialism"
            },
            {
                "reference": "Proverbs 13:11",
                "text": "Dishonest money dwindles away, but whoever gathers money little by little makes it grow.",
                "context": "Honest wealth-building vs shortcuts"
            },
            {
                "reference": "Luke 12:15",
                "text": "Then he said to them, 'Watch out! Be on your guard against all kinds of greed; life does not consist in an abundance of possessions.'",
                "context": "Christ's warning against materialism"
            }
        ],
        "work": [
            {
                "reference": "Colossians 3:23",
                "text": "Whatever you do, work at it with all your heart, as working for the Lord, not for human masters.",
                "context": "Excellence and dedication in all work"
            },
            {
                "reference": "2 Thessalonians 3:10",
                "text": "For even when we were with you, we gave you this rule: 'The one who is unwilling to work shall not eat.'",
                "context": "Biblical work ethic and responsibility"
            }
        ],
        "freedom": [
            {
                "reference": "Galatians 5:1",
                "text": "It is for freedom that Christ has set us free. Stand firm, then, and do not let yourselves be burdened again by a yoke of slavery.",
                "context": "Christ's gift of freedom and responsibility"
            },
            {
                "reference": "2 Corinthians 3:17",
                "text": "Now the Lord is the Spirit, and where the Spirit of the Lord is, there is freedom.",
                "context": "True freedom found in God"
            }
        ],
        "peace": [
            {
                "reference": "Matthew 5:9",
                "text": "Blessed are the peacemakers, for they will be called children of God.",
                "context": "Christ's blessing on those who pursue peace"
            },
            {
                "reference": "Romans 12:18",
                "text": "If it is possible, as far as it depends on you, live at peace with everyone.",
                "context": "Personal responsibility for pursuing peace"
            }
        ]
    }
    
    @classmethod
    def get_verses_for_topic(cls, topic: str) -> Optional[List[Dict[str, str]]]:
        """
        Get verses for a specific topic
        
        Args:
            topic: The topic to search for (e.g., 'immigration', 'justice')
            
        Returns:
            List of verse dictionaries with reference, text, and context
            Returns None if topic not found
        """
        return cls.VERSES.get(topic.lower())
    
    @classmethod
    def find_relevant_verses(cls, keywords: List[str]) -> List[Dict[str, str]]:
        """
        Find verses matching any of the provided keywords
        
        Args:
            keywords: List of keywords to search for (e.g., ['justice', 'compassion'])
            
        Returns:
            List of verse dictionaries (may contain duplicates if multiple keywords match)
        """
        relevant_verses = []
        for keyword in keywords:
            verses = cls.get_verses_for_topic(keyword.lower())
            if verses:
                relevant_verses.extend(verses)
        return relevant_verses
    
    @classmethod
    def get_all_topics(cls) -> List[str]:
        """Get list of all available topics"""
        return list(cls.VERSES.keys())
