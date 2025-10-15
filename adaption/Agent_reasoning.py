"""
Agent_reasoning.py
==============================

A CLI tool that analyzes academic problem solutions in JSONL format and extracts
structured, progressive hints using an LLM. The tool has two main modes:

1. **JSONL Processing**: Extract hints from solution traces and output structured JSON
2. **TSV Expansion**: Take existing TSV datasets and expand each question into multiple 
   easier variants by adding hints step-by-step (逐层地), creating progressive difficulty levels

Features
--------
- Logic tree construction from solution steps
- Multi-tier hint generation (minimal/standard/guided difficulty)
- Progressive hint expansion with granular control (hint-by-hint or tier-by-tier)
- Answer consistency validation between JSONL and TSV data
- Deterministic output with stable sorting
- Parallel processing support with configurable worker threads
- Optimized LLM API calls with concurrent processing at both record and variant levels

Usage Examples
-------------
# Basic JSONL hint extraction
python Agent_reasoning.py --in academic_problems.jsonl

# With explicit output and English translation  
python Agent_reasoning.py --in problems.jsonl --out my_hints.jsonl --force-english

# With parallel processing using multiple workers
python Agent_reasoning.py --in problems.jsonl --num-workers 4

# Using environment variables for API credentials
MODEL_API_KEY=my_key MODEL_API_BASE_URL=https://api.example.com/v1 python Agent_reasoning.py --in problems.jsonl

# Dry run to test with 5 records and parallel processing
python Agent_reasoning.py --in problems.jsonl --dry-run 5 --num-workers 2

# TSV expansion: create progressive hint variants
python Agent_reasoning.py --in solutions.jsonl --tsv-in data.tsv --tsv-out data_expanded.tsv --granularity hint

# TSV expansion with hard evolvement and custom misleading versions count
python Agent_reasoning.py --in solutions.jsonl --tsv-in data.tsv --tsv-out data_expanded.tsv --evolvement hard --hard_num 6 --num-workers 4

# With custom model parameters and parallel processing
python Agent_reasoning.py --in problems.jsonl --temperature 0.2 --top-p 0.95 --seed 42 --num-workers 4

# With hard evolvement and custom number of misleading versions
python Agent_reasoning.py --in problems.jsonl --evolvement hard --hard_num 6 --num-workers 3

TSV Expansion Mode
------------------
When --tsv-in and --tsv-out are provided, the tool operates in TSV expansion mode:

- Reads a TSV file with columns: index, question, hint, answer, category, task_type, sub_task_types, split, image_path
- Processes corresponding JSONL records to extract progressive hints
- Creates multiple variants of each question with cumulative hints (逐层地)
- Outputs expanded TSV with same schema, incremental numeric indices (e.g., 123, 124, 125, ...)
- Each expanded row contains all previous hints plus new ones in the hint column
- Supports two granularity levels:
  * `hint`: Add one hint at a time (hint1, hint1+hint2, hint1+hint2+hint3, ...)
  * `tier`: Add one tier at a time (tier1, tier1+tier2, tier1+tier2+tier3, ...)
- Preserves original question and answer in all variants
- Maintains deterministic order and prevents spoiler leakage
- **PERFORMANCE**: Utilizes parallel processing with --num-workers for significantly faster LLM API calls
- **SCALABILITY**: Automatically switches between sequential and parallel processing based on num_workers setting
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
# Mute OpenAI HTTP request logs - comprehensive approach
logging.getLogger("openai._client").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)  
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# More aggressive: disable HTTP logs from any _client module
logging.getLogger().addFilter(lambda record: not ('HTTP Request:' in str(record.getMessage()) or '_client.py' in record.pathname))

logger = logging.getLogger(__name__)

# Pydantic Data Models
class LogicNode(BaseModel):
    id: str
    kind: str  # "given" | "construction" | "observation" | "lemma" | "goal"
    text: str
    from_steps: List[int] = Field(default_factory=list)

class LogicEdge(BaseModel):
    src: str
    dst: str
    relation: str = "supports"  # or "derives"

class LogicTree(BaseModel):
    nodes: List[LogicNode] = Field(default_factory=list)
    edges: List[LogicEdge] = Field(default_factory=list)

class HintUnit(BaseModel):
    text: str
    from_infer_indices: List[int] = Field(default_factory=list)

class HintTier(BaseModel):
    tier: int
    label: str
    hints: List[HintUnit] = Field(default_factory=list)

class HintSet(BaseModel):
    difficulty: str  # minimal | standard | guided
    variant_id: int
    tiers: List[HintTier]

class ExtractionResult(BaseModel):
    index: int
    logic_tree: Optional[LogicTree] = None
    hint_sets: List[HintSet] = Field(default_factory=list)
    technique_tags: List[str] = Field(default_factory=list)
    rationale: str = ""
    echo_user_instruction: str = ""
    sanity_checks: Dict[str, Any] = Field(default_factory=dict)

def init_client(api_key, base_url):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client

def read_tsv(tsv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Read TSV file and return a dict keyed by index.
    
    Expected columns: index, question, hint, answer, category, task_type, sub_task_types, split, image_path
    
    Args:
        tsv_path: Path to the TSV file
        
    Returns:
        Dict mapping index (as string) to row data dict
    """
    tsv_data = {}
    
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate expected columns
            expected_cols = {'index', 'question', 'hint', 'answer', 'category', 'task_type', 'sub_task_types', 'split', 'image_path'}
            if not expected_cols.issubset(set(reader.fieldnames or [])):
                missing = expected_cols - set(reader.fieldnames or [])
                logger.warning(f"Missing expected columns in TSV: {missing}")
            
            for row in reader:
                # Use index as string key for safety
                index_key = str(row.get('index', ''))
                if index_key:
                    tsv_data[index_key] = row
                else:
                    logger.warning(f"Skipping row with empty index: {row}")
                    
    except Exception as e:
        logger.error(f"Error reading TSV file {tsv_path}: {e}")
        raise
    
    logger.info(f"Read {len(tsv_data)} rows from {tsv_path}")
    return tsv_data

def write_tsv(tsv_path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    """
    Write rows to TSV file with specified column order.
    
    Args:
        tsv_path: Output TSV file path
        rows: List of row dictionaries
        fieldnames: Column names in desired order
    """
    try:
        with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
            
        logger.info(f"Wrote {len(rows)} rows to {tsv_path}")
        
    except Exception as e:
        logger.error(f"Error writing TSV file {tsv_path}: {e}")
        raise

class GeoHintExtractor:
    """
    Extracts layered academic hints from solution traces using an LLM.
    """
    # Threshold for considering inference process too short
    SHORT_INFERENCE_THRESHOLD = 3
    
    def __init__(
        self, 
        client, 
        *, 
        max_hint_chars: int = 220,
        force_english: bool = False, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        seed: int = 0,
        model: str = "gpt-4",
        difficulty: str = "standard",
        variants: int = 1,
        max_tiers: int = 3,
        style: str = "direct",
        timeout: float = 60.0,
        evolvement: str = "easy",
        num_workers: int = 1,
        hard_num: int = 4,
        expand_size: int = 4
    ):
        """
        Initialize the extractor with an LLM client and configuration.
        
        Args:
            client: The LLM API client
            max_hint_chars: Maximum characters per hint
            force_english: Whether to translate hints to English
            temperature: LLM temperature parameter (0.0 = deterministic)
            top_p: LLM top-p parameter (1.0 = no filtering)
            seed: Random seed for deterministic generation
            model: LLM model name to use
            difficulty: Hint difficulty level ("minimal", "standard", "guided")
            variants: Number of hint variants to generate
            max_tiers: Maximum number of hint tiers
            style: Hint style ("socratic" or "direct")
            timeout: Timeout for LLM calls
            evolvement: Evolvement type ("easy", "hard", or "easy,hard")
            num_workers: Number of worker threads for parallel LLM API calls
            hard_num: Number of misleading versions to generate for hard evolvement
        """
        self.client = client
        self.max_hint_chars = max_hint_chars
        self.force_english = force_english
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.model = model
        self.difficulty = difficulty
        self.variants = variants
        self.max_tiers = max_tiers
        self.style = style
        self.timeout = timeout
        self.evolvement = evolvement
        self.num_workers = num_workers
        self.hard_num = hard_num
        self.expand_size = expand_size
        
        # Tracking metrics
        self.processed_count = 0
        self.failed_count = 0
        self.emitted_tiers_total = 0

    # Illustration:
    # extractor = GeoHintExtractor(
    #     client=init_client(api_key, base_url),
    #     max_hint_chars=220,
    #     force_english=False,
    #     temperature=0.0,
    #     top_p=1.0,
    #     seed=0
    # )

    def call_llm_json(self, messages: List[Dict], expect: str = "object") -> Union[Dict, List]:
        """
        Call the LLM with timeout, retries, and robust JSON parsing.
        
        Args:
            messages: List of message dictionaries for the LLM
            expect: Expected return type - "object" or "array"
            
        Returns:
            Parsed JSON response (dict or list)
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed if self.seed != 0 else None,
                    timeout=self.timeout
                )
                
                # Extract the content from the response
                response_text = response.choices[0].message.content
                if not response_text:
                    logger.warning(f"Empty response from LLM (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        return {} if expect == "object" else []
                    continue
                
                response_text = response_text.strip()
                
                # Handle different response formats
                if isinstance(response_text, dict) and 'text' in response_text:
                    response_text = response_text['text']
                elif not isinstance(response_text, str):
                    logger.error(f"Unexpected response format: {type(response_text)}")
                    return {} if expect == "object" else []
                
                # Check if response is empty after stripping
                if not response_text:
                    logger.warning(f"Empty response content after stripping (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        return {} if expect == "object" else []
                    continue
                
                # Robust JSON parsing with math notation handling
                try:
                    # Clean up response text - remove markdown fences
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].strip()
                    
                    # Find JSON boundaries if needed
                    if not (response_text.startswith('{') or response_text.startswith('[')):
                        # Look for first { or [
                        start_pos = min((response_text.find('{'), response_text.find('[')), 
                                      key=lambda x: float('inf') if x == -1 else x)
                        if start_pos != -1:
                            response_text = response_text[start_pos:]
                    
                    # Handle mathematical notation by escaping backslashes properly
                    # First, temporarily replace escaped quotes to avoid conflicts
                    response_text = response_text.replace('\\"', '<<ESCAPED_QUOTE>>')
                    
                    # Fix common mathematical escape sequences that cause JSON parsing errors
                    math_fixes = {
                        r'\delta': r'\\delta',
                        r'\sin': r'\\sin',
                        r'\cos': r'\\cos', 
                        r'\tan': r'\\tan',
                        r'\pi': r'\\pi',
                        r'\alpha': r'\\alpha',
                        r'\beta': r'\\beta',
                        r'\gamma': r'\\gamma',
                        r'\theta': r'\\theta',
                        r'\lambda': r'\\lambda',
                        r'\mu': r'\\mu',
                        r'\sigma': r'\\sigma',
                        r'\omega': r'\\omega',
                        r'\Omega': r'\\Omega',
                        r'\mathbb': r'\\mathbb',
                        r'\overrightarrow': r'\\overrightarrow',
                        r'\overleftarrow': r'\\overleftarrow',
                        r'\vec': r'\\vec',
                        r'\cdot': r'\\cdot',
                        r'\times': r'\\times',
                        r'\frac': r'\\frac',
                        r'\sqrt': r'\\sqrt',
                        r'\sum': r'\\sum',
                        r'\int': r'\\int',
                        r'\lim': r'\\lim',
                        r'\infty': r'\\infty',
                        r'\partial': r'\\partial',
                        r'\nabla': r'\\nabla',
                        r'\in': r'\\in',
                        r'\subset': r'\\subset',
                        r'\cup': r'\\cup',
                        r'\cap': r'\\cap',
                        r'\rightarrow': r'\\rightarrow',
                        r'\leftarrow': r'\\leftarrow',
                        r'\Rightarrow': r'\\Rightarrow',
                        r'\Leftarrow': r'\\Leftarrow',
                        r'\leftrightarrow': r'\\leftrightarrow',
                        r'\Leftrightarrow': r'\\Leftrightarrow',
                        r'\leq': r'\\leq',
                        r'\geq': r'\\geq',
                        r'\neq': r'\\neq',
                        r'\approx': r'\\approx',
                        r'\equiv': r'\\equiv',
                        r'\pm': r'\\pm',
                        r'\mp': r'\\mp',
                        r'\log': r'\\log',
                        r'\ln': r'\\ln',
                        r'\exp': r'\\exp',
                        # Control characters and problematic sequences
                        r'\t': r'\\t',
                        r'\n': r'\\n',
                        r'\r': r'\\r',
                        # Additional Greek letters and symbols from error logs
                        r'\eta': r'\\eta',
                        r'\kappa': r'\\kappa',
                        r'\rho': r'\\rho',
                        r'\tau': r'\\tau',
                        r'\phi': r'\\phi',
                        r'\chi': r'\\chi',
                        r'\psi': r'\\psi',
                        r'\zeta': r'\\zeta',
                        r'\xi': r'\\xi',
                        r'\nu': r'\\nu',
                        r'\iota': r'\\iota',
                        r'\upsilon': r'\\upsilon',
                        # Brackets and parentheses
                        r'\{': r'\\{',
                        r'\}': r'\\}',
                        r'\[': r'\\[',
                        r'\]': r'\\]',
                        r'\(': r'\\(',
                        r'\)': r'\\)',
                        # Additional LaTeX commands
                        r'\text': r'\\text',
                        r'\mathrm': r'\\mathrm',
                        r'\mathit': r'\\mathit',
                        r'\mathbf': r'\\mathbf',
                        r'\mathcal': r'\\mathcal',
                        r'\left': r'\\left',
                        r'\right': r'\\right',
                        r'\big': r'\\big',
                        r'\Big': r'\\Big',
                        r'\bigg': r'\\bigg',
                        r'\Bigg': r'\\Bigg',
                        # Add more common math symbols as needed
                    }
                    
                    # Apply fixes only within string values (between quotes)
                    import re
                    def fix_math_in_strings(match):
                        string_content = match.group(1)
                        for pattern, replacement in math_fixes.items():
                            # Only replace if it's not already escaped
                            string_content = re.sub(f'(?<!\\\\){re.escape(pattern)}', replacement, string_content)
                        return f'"{string_content}"'
                    
                    # Find and fix math notation in JSON string values
                    response_text = re.sub(r'"([^"]*)"', fix_math_in_strings, response_text)
                    
                    # Restore escaped quotes
                    response_text = response_text.replace('<<ESCAPED_QUOTE>>', '\\"')
                    
                    result = json.loads(response_text)
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {e}")
                    logger.debug(f"Raw response (first 500 chars): {response_text[:500]}")
                    
                    # Try more aggressive fixes for common issues
                    if attempt == max_retries - 1:
                        logger.info("Attempting aggressive JSON repair...")
                        repaired_json = self._repair_json_aggressively(response_text, expect)
                        if repaired_json is not None:
                            logger.info("Successfully repaired JSON")
                            return repaired_json
                        
                        logger.error(f"JSON parsing failed after all retries. Full response: {response_text}")
                        return {} if expect == "object" else []
                    continue
                    
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {} if expect == "object" else []
                time.sleep(1)  # Brief delay before retry
        
        return {} if expect == "object" else []

    def _repair_json_aggressively(self, response_text: str, expect: str) -> Union[Dict, List, None]:
        """
        Attempt aggressive JSON repair for mathematical content.
        
        Args:
            response_text: The problematic JSON text
            expect: Expected return type - "object" or "array"
            
        Returns:
            Repaired JSON object/array or None if repair fails
        """
        try:
            # Strategy 1: Replace all single backslashes with double backslashes
            repaired = response_text
            
            # Find all positions of backslashes that aren't already escaped
            import re
            
            # Replace single backslashes with double, but be careful with already escaped ones
            # This is a more aggressive approach
            repaired = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', repaired)
            
            # Try parsing
            try:
                result = json.loads(repaired)
                return result
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Remove problematic control characters
            repaired = response_text
            # Remove actual control characters (not escaped ones)
            repaired = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', repaired)
            
            # Fix common escape issues by replacing with safe alternatives
            problematic_escapes = {
                r'\delta': 'delta',
                r'\sin': 'sin',
                r'\cos': 'cos',
                r'\tan': 'tan',
                r'\pi': 'pi',
                r'\alpha': 'alpha',
                r'\beta': 'beta',
                r'\gamma': 'gamma',
                r'\theta': 'theta',
                r'\lambda': 'lambda',
                r'\mu': 'mu',
                r'\sigma': 'sigma',
                r'\omega': 'omega',
                r'\Omega': 'Omega',
                r'\overrightarrow': 'vector',
                r'\overleftarrow': 'vector',
                r'\vec': 'vector',
                r'\mathbb': '',
                r'\frac': 'fraction',
                r'\sqrt': 'sqrt',
                r'\sum': 'sum',
                r'\int': 'integral',
                r'\lim': 'limit',
                r'\infty': 'infinity',
                r'\in': 'in',
                r'\subset': 'subset',
                r'\cup': 'union',
                r'\cap': 'intersection',
                r'\rightarrow': 'to',
                r'\leftarrow': 'from',
                r'\leq': 'leq',
                r'\geq': 'geq',
                r'\neq': 'neq',
                r'\approx': 'approx',
                r'\equiv': 'equiv',
                r'\pm': 'plus_minus',
                r'\cdot': '*',
                r'\times': '*',
                r'\log': 'log',
                r'\ln': 'ln',
                r'\exp': 'exp',
            }
            
            # Apply replacements only within quoted strings
            def replace_math_symbols(match):
                string_content = match.group(1)
                for pattern, replacement in problematic_escapes.items():
                    string_content = string_content.replace(pattern, replacement)
                return f'"{string_content}"'
            
            repaired = re.sub(r'"([^"]*)"', replace_math_symbols, repaired)
            
            try:
                result = json.loads(repaired)
                return result
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Extract content manually using regex
            if expect == "array":
                # Try to extract array elements manually
                array_match = re.search(r'\[(.*)\]', repaired, re.DOTALL)
                if array_match:
                    array_content = array_match.group(1)
                    
                    # Try to extract version/hints objects
                    version_objects = []
                    
                    # Look for version and hints patterns
                    version_pattern = r'"version":\s*(\d+).*?"hints":\s*\[(.*?)\]'
                    for match in re.finditer(version_pattern, array_content, re.DOTALL):
                        version = int(match.group(1))
                        hints_str = match.group(2)
                        
                        # Extract individual hints
                        hints = []
                        hint_matches = re.findall(r'"([^"]*)"', hints_str)
                        for hint in hint_matches:
                            # Clean up the hint
                            clean_hint = hint.replace('\\\\', '\\').strip()
                            if clean_hint:
                                hints.append(clean_hint)
                        
                        if hints:
                            version_objects.append({
                                "version": version,
                                "hints": hints
                            })
                    
                    if version_objects:
                        return version_objects
            
            # Strategy 4: Return minimal valid structure
            logger.warning("All JSON repair strategies failed, returning minimal structure")
            if expect == "array":
                return []
            else:
                return {}
                
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return None

    def make_wrong_hint(self, lemma_text: str) -> str:
        """Call LLM to convert lemma to misleading hint for hard evolvement."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"""Based on the following lemma, create a misleading hint for a student.  

Requirements:  
1. The hint must sound reasonable and stay related to the lemma.  
2. It should subtly direct the student toward applying an incorrect theorem, method, or idea that seems plausible but cannot solve the problem.  
3. Keep the hint concise and natural, as if a real tutor gave it.  
4. Avoid making the error too obvious or the hint completely irrelevant. 
5. The hint must not reveal or suggest that it is misleading, wrong, or a trick. It should read as if it is a genuine helpful hint.

Lemma:  
{lemma_text}

Misleading Hint:"""
                }],
                temperature=0.7,
                top_p=0.9,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate misleading hint: {e}")
            return f"Try using a different approach than: {lemma_text[:100]}..."

    def make_refined_hint(self, hint_text: str, mislead: bool = False) -> str:
        """Call LLM to refine existing hint - make it clearer or more misleading."""
        try:
            if not mislead:
                prompt = f"""Please refine the following hint by summarizing it into a shorter, clearer form.  
                    Requirements:  
                    1. Keep the content faithful to the original hint (do not add new ideas or remove essential reasoning).  
                    2. Make the language more concise and direct.  
                    3. Preserve all key information needed to understand the hint.  

                    Original Hint:  
                    {hint_text}

                    Refined Hint:"""
            else:
                prompt = f"""You will receive a piece of prompt text, referred to as the "misleading prompt," which will guide the problem solver toward incorrect reasoning. 
            Your task is: while keeping the length of the prompt roughly similar, rewrite it to be more misleading. 
            Requirements: 
            1. Do not add too much new information; the overall word count should be similar to the original text. 
            2. The errors should be more subtle, appearing reasonable but actually steering people away from the correct reasoning. 
            3. Keep it related to the context of the problem; do not write completely irrelevant statements. 
            4. The hint must not reveal or suggest that it is misleading, wrong, or a trick. It should read as if it is a genuine helpful hint.
            Input prompt: {hint_text} 
            Output prompt (more misleading version):"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4 if not mislead else 0.7,
                top_p=0.9,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to refine hint: {e}")
            return hint_text

    def generate_misleading_versions(self, record: dict, num_versions: int = 4) -> List[Dict[str, str]]:
        """Generate misleading versions of a problem for hard evolvement."""
        versions = []
        question = record.get('question', '')
        infer_process = record.get('infer_process', [])
        answer = record.get('answer', '')
        
        if not question:
            logger.warning("Empty question for misleading version generation")
            return [{"version": 0, "text": question}]
        
        # Version 0: Original problem
        versions.append({"version": 0, "text": question})
        
        try:
            # Check if inference process is too short - use Q&A-based generation
            if not infer_process or len(infer_process) < self.SHORT_INFERENCE_THRESHOLD:
                logger.info(f"Inference process too short ({len(infer_process)} steps < {self.SHORT_INFERENCE_THRESHOLD}) for misleading version generation, using Q&A-based approach")
                
                if answer:
                    # Use Q&A-based misleading hint generation
                    qa_versions = self.generate_misleading_hints_from_qa(question, answer, num_versions)
                    return qa_versions
                else:
                    logger.warning("No answer available for Q&A-based misleading hint generation")
                    # Fallback: return versions with just the original question
                    for i in range(1, num_versions):
                        versions.append({"version": i, "text": question})
                    return versions
            
            # Original logic for long infer_process
            # Use steps from infer_process as "lemmas" for misleading hint generation
            # Treat each inference step as a potential source for misleading hints
            step = max(1, len(infer_process) // (num_versions - 1)) if num_versions > 1 else 1
            
            for i in range(1, num_versions):
                selected_steps = infer_process[:i * step]
                if not selected_steps:
                    continue
                    
                # Generate misleading hints from selected steps
                misleading_hints = []
                for step_text in selected_steps:
                    if len(step_text.strip()) > 10:  # Only use substantial steps
                        try:
                            misleading_hint = self.make_wrong_hint(step_text)
                            misleading_hints.append(misleading_hint)
                        except Exception as e:
                            logger.warning(f"Failed to generate misleading hint from step: {e}")
                            continue
                
                if misleading_hints:
                    # Create combined misleading hint text
                    hint_text = "Hints:\n" + "\n".join(f"- {h}" for h in misleading_hints)
                    
                    # Refine the combined hints to make them more subtly misleading
                    try:
                        hint_text = self.make_refined_hint(hint_text, mislead=True)
                    except Exception as e:
                        logger.warning(f"Failed to refine misleading hints: {e}")
                    
                    versions.append({
                        "version": i,
                        "text": question + "\n\n" + hint_text
                    })
                else:
                    # Fallback: just add original question
                    versions.append({"version": i, "text": question})
                    
        except Exception as e:
            logger.error(f"Error generating misleading versions: {e}")
            # Return just the original version on error
            return [{"version": 0, "text": question}]
        
        return versions

    def build_llm_prompt(self, infer_process: list[str], force_english: bool, user_instruction: str) -> str:
        """
        Build a focused prompt for the LLM to extract just the hints.
        
        Args:
            infer_process: List of reasoning steps
            force_english: Whether to translate hints to English
            user_instruction: Optional user guidance
            
        Returns:
            A concise prompt string for the LLM
        """
        prompt = f"""
    Analyze these academic problem solution steps and extract key hints that would help a student solve the problem.

    # Solution steps:
    {json.dumps(infer_process, indent=2)}

    # Your task:
    1. Identify 1-3 tiers of helpful hints, from basic orientation to key insights
    2. For each hint, specify which step numbers (0-based indices) it comes from

    # Focus on:
    - Key insights or observations
    - Critical techniques or methods
    - Important formulas or theorems
    - Progressive revelation (simpler hints first)
    - Being concise and actionable
    {"- Translate hints to English" if force_english else "- Preserve the original language"}

    {f"# User guidance: {user_instruction}" if user_instruction else ""}

    # Output format:
    Return a JSON array of hint tiers like this:
    [
    {{
        "tier_label": "Initial insight",
        "hints": [
        {{"hint": "Apply the key theorem or formula from step 2", "from_steps": [2, 3]}},
        {{"hint": "Consider the relationship established in step 4", "from_steps": [4]}}
        ]
    }},
    {{
        "tier_label": "Key observation",
        "hints": [
        {{"hint": "Notice that triangles ABC and DEF are similar", "from_steps": [5, 6]}}
        ]
    }}
    ]

    If the solution is simple, a single tier is sufficient. Be concise and precise.
    """
        return prompt

    # Illustration:
    # infer_process = ["Let the curve intersect the sphere at points \\(A\\) and \\(B\\).",
    #                  "Let \\(B'\\) be the point diametrically opposite \\(A\\).",
    #                  "We claim that the curve must lie in the hemisphere."]
    # user_instruction = "Focus on auxiliary constructions"
    # prompt = extractor.build_llm_prompt(infer_process, False, user_instruction)
    # The prompt includes the task description, rules, input steps in JSON format,
    # parameters (force_english=false), the user_instruction, and the expected output format.

    def build_logic_tree(self, infer_process: List[str]) -> LogicTree:
        """
        Build a logic tree (DAG) from inference steps using LLM + heuristics.
        
        Args:
            infer_process: List of solution steps
            
        Returns:
            LogicTree with nodes and edges
        """
        if not infer_process:
            return LogicTree()
        
        # Heuristic pass: identify constructions with regex
        construction_patterns = [
            r"intersection of", r"perpendicular", r"midpoint", r"reflection",
            r"bisector", r"circumcircle", r"parallel", r"tangent", r"homothety",
            r"inversion", r"construct", r"draw", r"let.*be.*point"
        ]
        
        seed_nodes = []
        for i, step in enumerate(infer_process):
            step_lower = step.lower()
            node_kind = "observation"  # default
            
            # Check for constructions
            if any(pattern in step_lower for pattern in construction_patterns):
                node_kind = "construction"
            # Check for goals/results
            elif any(word in step_lower for word in ["therefore", "thus", "hence", "qed", "equals"]):
                node_kind = "goal"
            # Check for givens
            elif i == 0 or any(word in step_lower for word in ["given", "let", "suppose"]):
                node_kind = "given"
            
            seed_nodes.append({
                "id": f"N{i+1}",
                "kind": node_kind,
                "text": step[:200],  # Truncate for brevity
                "from_steps": [i]
            })
        
        # LLM prompt for logic tree
        logic_tree_prompt = f"""You will transform a list of geometry proof steps into a logic tree (DAG) of subgoals/facts.

# Steps (0-based indices):
{json.dumps(infer_process, indent=2)}

# Output strictly as JSON with keys: nodes, edges.
Schema:
{{
  "nodes": [
    {{"id": "N1", "kind": "given|construction|observation|lemma|goal", "text": "...", "from_steps": [0,2]}},
    ...
  ],
  "edges": [
    {{"src": "N1", "dst": "N3", "relation": "supports"}},
    ...
  ]
}}

Rules:
- Prefer few, meaningful nodes (7–15 typical).
- Create "goal" as the final target claim.
- Map constructions (perpendiculars, midpoints, reflections, circumcircles, intersections) to "construction".
- Use "lemma" for pivotal facts (e.g., similarity, cyclic).
- Fill from_steps with the step indices that first introduce/support the node.
- Ensure DAG (no cycles). No explanations outside JSON."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": logic_tree_prompt}
        ]
        
        try:
            llm_result = self.call_llm_json(messages, expect="object")
            
            # Merge LLM result with heuristics
            nodes = []
            edges = []
            
            if "nodes" in llm_result and isinstance(llm_result["nodes"], list):
                for node_data in llm_result["nodes"]:
                    if isinstance(node_data, dict):
                        node = LogicNode(
                            id=node_data.get("id", f"N{len(nodes)+1}"),
                            kind=node_data.get("kind", "observation"),
                            text=node_data.get("text", ""),
                            from_steps=node_data.get("from_steps", [])
                        )
                        nodes.append(node)
            
            if "edges" in llm_result and isinstance(llm_result["edges"], list):
                for edge_data in llm_result["edges"]:
                    if isinstance(edge_data, dict):
                        edge = LogicEdge(
                            src=edge_data.get("src", ""),
                            dst=edge_data.get("dst", ""),
                            relation=edge_data.get("relation", "supports")
                        )
                        edges.append(edge)
            
            # Fallback: if LLM failed, use heuristic nodes
            if not nodes:
                for node_data in seed_nodes[:10]:  # Limit to reasonable size
                    node = LogicNode(**node_data)
                    nodes.append(node)
                
                # Create simple chain edges
                for i in range(len(nodes) - 1):
                    edge = LogicEdge(src=nodes[i].id, dst=nodes[i+1].id)
                    edges.append(edge)
            
            # Validate DAG (basic check - no self-loops)
            valid_edges = []
            for edge in edges:
                if edge.src != edge.dst:
                    valid_edges.append(edge)
            
            return LogicTree(nodes=nodes, edges=valid_edges)
            
        except Exception as e:
            logger.warning(f"Logic tree extraction failed: {e}")
            # Return heuristic fallback
            nodes = [LogicNode(**node_data) for node_data in seed_nodes[:10]]
            edges = []
            for i in range(len(nodes) - 1):
                edge = LogicEdge(src=nodes[i].id, dst=nodes[i+1].id)
                edges.append(edge)
            return LogicTree(nodes=nodes, edges=edges)

    def plan_hint_tiers(self, logic_tree: LogicTree, difficulty: str, max_tiers: int, style: str) -> List[HintTier]:
        """
        Plan progressive hint tiers based on logic tree and difficulty level.
        
        Args:
            logic_tree: The logic tree from build_logic_tree
            difficulty: "minimal", "standard", or "guided"  
            max_tiers: Maximum number of tiers to create
            style: "socratic" (questions) or "direct" (statements)
            
        Returns:
            List of HintTier objects
        """
        if not logic_tree.nodes:
            return []
        
        # Create the hint planning prompt
        hint_plan_prompt = f"""Using this logic tree, plan progressive, revealable hints for the chosen difficulty.

# Logic tree (JSON):
{logic_tree.model_dump_json(indent=2)}

# Constraints:
- difficulty={difficulty}; style={style}; max_tiers={max_tiers}
- Tiers progress from broad orientation to key insight. No full solution.
- Each hint links to source step indices via from_steps.
- Avoid final result equations/values; name constructions/lemmas only when allowed by difficulty.
- If style=socratic, phrase mostly as questions.

# Difficulty rules:
- minimal: 1–2 tiers, very high-level; no theorem names if they reveal the approach directly; focus on "where to look".
- standard: 2–3 tiers; include named constructions and non-final lemmas; avoid explicit final similarity/equality unless absolutely needed.
- guided: up to {max_tiers}; include key lemmas and the pivotal construction; still avoid the final equality/result.

# Output JSON:
[
  {{
    "tier": 1,
    "label": "Initial orientation",
    "hints": [
      {{"text": "…", "from_steps": [0,2]}},
      ...
    ]
  }},
  ...
]"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": hint_plan_prompt}
        ]
        
        try:
            llm_result = self.call_llm_json(messages, expect="array")
            
            tiers = []
            if isinstance(llm_result, list):
                for i, tier_data in enumerate(llm_result[:max_tiers]):
                    if not isinstance(tier_data, dict):
                        continue
                    
                    hints = []
                    for hint_data in tier_data.get("hints", []):
                        if isinstance(hint_data, dict) and hint_data.get("text"):
                            hint = HintUnit(
                                text=hint_data.get("text", ""),
                                from_infer_indices=hint_data.get("from_steps", [0])
                            )
                            hints.append(hint)
                    
                    if hints:
                        tier = HintTier(
                            tier=i + 1,
                            label=tier_data.get("label", f"Tier {i + 1}"),
                            hints=hints
                        )
                        tiers.append(tier)
            
            # Fallback: create basic tiers from logic tree nodes
            if not tiers and logic_tree.nodes:
                tiers = self._create_fallback_tiers(logic_tree, difficulty, max_tiers, style)
            
            return tiers
            
        except Exception as e:
            logger.warning(f"Hint tier planning failed: {e}")
            return self._create_fallback_tiers(logic_tree, difficulty, max_tiers, style)

    def _create_fallback_tiers(self, logic_tree: LogicTree, difficulty: str, max_tiers: int, style: str) -> List[HintTier]:
        """Create fallback tiers from logic tree when LLM fails."""
        tiers = []
        
        # Group nodes by kind for different tiers
        constructions = [n for n in logic_tree.nodes if n.kind == "construction"]
        lemmas = [n for n in logic_tree.nodes if n.kind == "lemma"]
        observations = [n for n in logic_tree.nodes if n.kind == "observation"]
        
        # Tier 1: Initial orientation
        tier1_hints = []
        if constructions and difficulty in ["standard", "guided"]:
            hint_text = f"Consider the construction mentioned in the proof."
            if style == "socratic":
                hint_text = "What key construction is introduced in the solution?"
            tier1_hints.append(HintUnit(text=hint_text, from_infer_indices=constructions[0].from_steps))
        elif observations:
            hint_text = f"Focus on the initial setup and given conditions."
            if style == "socratic":
                hint_text = "What relationships are given or immediately apparent?"
            tier1_hints.append(HintUnit(text=hint_text, from_infer_indices=observations[0].from_steps))
        
        if tier1_hints:
            tiers.append(HintTier(tier=1, label="Initial orientation", hints=tier1_hints))
        
        # Tier 2: Key insight (for standard/guided)
        if len(tiers) < max_tiers and difficulty in ["standard", "guided"]:
            tier2_hints = []
            if lemmas:
                hint_text = f"Look for key geometric relationships or similarities."
                if style == "socratic":
                    hint_text = "What important theorem or relationship applies here?"
                tier2_hints.append(HintUnit(text=hint_text, from_infer_indices=lemmas[0].from_steps))
            elif len(constructions) > 1:
                hint_text = f"Consider how the constructions work together."
                if style == "socratic":
                    hint_text = "How do the different constructions relate to each other?"
                tier2_hints.append(HintUnit(text=hint_text, from_infer_indices=constructions[1].from_steps))
            
            if tier2_hints:
                tiers.append(HintTier(tier=2, label="Key insight", hints=tier2_hints))
        
        return tiers[:max_tiers]

    def sample_variants(self, tiers: List[HintTier], variants: int) -> List[List[HintTier]]:
        """
        Generate multiple variants of hint tiers through paraphrasing.
        
        Args:
            tiers: Original list of hint tiers
            variants: Number of variants to generate
            
        Returns:
            List of tier lists (each representing a variant)
        """
        if variants <= 1 or not tiers:
            return [tiers]
        
        # Convert tiers to dict format for LLM
        tiers_data = []
        for tier in tiers:
            tier_dict = {
                "tier": tier.tier,
                "label": tier.label,
                "hints": [{"text": hint.text, "from_steps": hint.from_infer_indices} 
                         for hint in tier.hints]
            }
            tiers_data.append(tier_dict)
        
        variant_prompt = f"""Paraphrase these hint tiers into {variants} alternative versions with the same tier count and meaning.
Keep from_steps mapped faithfully.

# Input tiers JSON:
{json.dumps(tiers_data, indent=2)}

# Output JSON:
[
  [{{"tier": 1, "label": "...", "hints": [...]}}, ...],  // variant 1
  [{{"tier": 1, "label": "...", "hints": [...]}}, ...]   // variant 2
]"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": variant_prompt}
        ]
        
        try:
            # Use slightly higher temperature for variants to encourage diversity
            original_temp = self.temperature
            self.temperature = min(0.4, max(self.temperature, 0.15))
            
            llm_result = self.call_llm_json(messages, expect="array")
            
            # Restore original temperature
            self.temperature = original_temp
            
            variant_lists = []
            if isinstance(llm_result, list):
                for variant_data in llm_result[:variants]:
                    if isinstance(variant_data, list):
                        variant_tiers = []
                        for tier_data in variant_data:
                            if isinstance(tier_data, dict):
                                hints = []
                                for hint_data in tier_data.get("hints", []):
                                    if isinstance(hint_data, dict) and hint_data.get("text"):
                                        hint = HintUnit(
                                            text=hint_data.get("text", ""),
                                            from_infer_indices=hint_data.get("from_steps", [0])
                                        )
                                        hints.append(hint)
                                
                                if hints:
                                    tier = HintTier(
                                        tier=tier_data.get("tier", len(variant_tiers) + 1),
                                        label=tier_data.get("label", f"Tier {len(variant_tiers) + 1}"),
                                        hints=hints
                                    )
                                    variant_tiers.append(tier)
                        
                        if variant_tiers:
                            variant_lists.append(variant_tiers)
            
            # Fallback: if we don't have enough variants, pad with slight modifications
            while len(variant_lists) < variants:
                if variant_lists:
                    # Use the first variant as template
                    base_variant = variant_lists[0]
                else:
                    # Use original tiers
                    base_variant = tiers
                
                # Create simple variant by modifying labels
                modified_variant = []
                for tier in base_variant:
                    new_hints = []
                    for hint in tier.hints:
                        # Simple paraphrasing fallback
                        new_text = hint.text
                        if new_text.endswith('.'):
                            new_text = new_text[:-1] + " in this problem."
                        elif new_text.endswith('?'):
                            new_text = "Consider: " + new_text[0].lower() + new_text[1:]
                        
                        new_hint = HintUnit(text=new_text, from_infer_indices=hint.from_infer_indices)
                        new_hints.append(new_hint)
                    
                    new_tier = HintTier(
                        tier=tier.tier,
                        label=tier.label + f" (variant {len(variant_lists) + 1})",
                        hints=new_hints
                    )
                    modified_variant.append(new_tier)
                
                variant_lists.append(modified_variant)
            
            return variant_lists[:variants]
            
        except Exception as e:
            logger.warning(f"Variant generation failed: {e}")
            # Return original tiers repeated
            return [tiers] * variants

    def self_critique_and_fix(self, tiers: List[HintTier], infer_len: int) -> List[HintTier]:
        """
        Self-critique and fix issues with hint tiers using LLM + local fixes.
        
        Args:
            tiers: List of hint tiers to review
            infer_len: Length of original inference process for index validation
            
        Returns:
            Fixed list of hint tiers
        """
        if not tiers:
            return tiers
        
        # Convert tiers to dict format for LLM
        tiers_data = []
        for tier in tiers:
            tier_dict = {
                "tier": tier.tier,
                "label": tier.label,
                "hints": [{"text": hint.text, "from_steps": hint.from_infer_indices} 
                         for hint in tier.hints]
            }
            tiers_data.append(tier_dict)
        
        critique_prompt = f"""Review hint tiers for (a) spoilers (final equality/result), (b) clarity, (c) index validity vs infer_len={infer_len}, (d) math markdown balance.
If issues, minimally edit text to fix them without adding information.

# Input tiers JSON:
{json.dumps(tiers_data, indent=2)}

# Output JSON: same structure, corrected."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": critique_prompt}
        ]
        
        try:
            llm_result = self.call_llm_json(messages, expect="array")
            
            fixed_tiers = []
            if isinstance(llm_result, list):
                for tier_data in llm_result:
                    if isinstance(tier_data, dict):
                        hints = []
                        for hint_data in tier_data.get("hints", []):
                            if isinstance(hint_data, dict) and hint_data.get("text"):
                                hint = HintUnit(
                                    text=hint_data.get("text", ""),
                                    from_infer_indices=hint_data.get("from_steps", [0])
                                )
                                hints.append(hint)
                        
                        if hints:
                            tier = HintTier(
                                tier=tier_data.get("tier", len(fixed_tiers) + 1),
                                label=tier_data.get("label", f"Tier {len(fixed_tiers) + 1}"),
                                hints=hints
                            )
                            fixed_tiers.append(tier)
            
            # If LLM failed, use original tiers
            if not fixed_tiers:
                fixed_tiers = tiers
                
        except Exception as e:
            logger.warning(f"Self-critique failed: {e}")
            fixed_tiers = tiers
        
        # Apply local fixes regardless of LLM success
        return self._apply_local_fixes(fixed_tiers, infer_len)

    def _apply_local_fixes(self, tiers: List[HintTier], infer_len: int) -> List[HintTier]:
        """Apply local fixes to hint tiers."""
        fixed_tiers = []
        
        for tier in tiers:
            fixed_hints = []
            for hint in tier.hints:
                # Fix indices - ensure they're valid
                valid_indices = [i for i in hint.from_infer_indices 
                               if isinstance(i, int) and 0 <= i < infer_len]
                if not valid_indices:
                    valid_indices = [0]  # Default to first step
                
                # Check for math markdown balance
                text = hint.text
                if text.count('\\(') != text.count('\\)'):
                    logger.warning(f"Unbalanced math markdown in hint: {text[:50]}...")
                    # Simple fix: remove unmatched delimiters
                    if text.count('\\(') > text.count('\\)'):
                        # Remove extra opening delimiters
                        parts = text.split('\\(')
                        needed_opens = text.count('\\)')
                        text = parts[0] + '\\('.join(parts[1:needed_opens+1])
                    else:
                        # Remove extra closing delimiters  
                        parts = text.split('\\)')
                        needed_closes = text.count('\\(')
                        text = '\\)'.join(parts[:needed_closes+1])
                
                # Check for spoilers (basic detection)
                spoiler_patterns = [
                    r'therefore.*equals?', r'thus.*=', r'hence.*=', 
                    r'qed', r'proved', r'result is', r'answer is'
                ]
                has_spoiler = any(re.search(pattern, text.lower()) for pattern in spoiler_patterns)
                if has_spoiler:
                    # Remove or soften spoiler language
                    text = re.sub(r'\btherefore\b', 'consider that', text, flags=re.IGNORECASE)
                    text = re.sub(r'\bthus\b', 'note that', text, flags=re.IGNORECASE)
                    text = re.sub(r'\bhence\b', 'observe that', text, flags=re.IGNORECASE)
                    text = re.sub(r'\bqed.*$', '', text, flags=re.IGNORECASE).strip()
                
                # Clamp text length
                if len(text) > self.max_hint_chars:
                    text = text[:self.max_hint_chars-3] + "..."
                
                if text.strip():
                    fixed_hint = HintUnit(text=text.strip(), from_infer_indices=valid_indices)
                    fixed_hints.append(fixed_hint)
            
            if fixed_hints:
                fixed_tier = HintTier(tier=tier.tier, label=tier.label, hints=fixed_hints)
                fixed_tiers.append(fixed_tier)
        
        return fixed_tiers

    def generate_hints_from_qa(self, question: str, answer: str, user_instruction: str = "") -> dict:
        """
        Generate hints directly from question and answer when inference process is too short.
        
        Args:
            question: The geometry problem question
            answer: The answer to the problem
            user_instruction: Optional user guidance
            
        Returns:
            A dictionary representation of ExtractionResult with generated hints
        """
        if not question or not answer:
            logger.warning("Empty question or answer for direct hint generation")
            return self._create_empty_extraction_result(0, user_instruction)
        
        try:
            # Create prompt for direct hint generation from Q&A
            qa_prompt = f"""You are an academic tutor. Based on the given question and answer, generate helpful hints that would guide a student to solve the problem step by step.

Question:
{question}

Answer:
{answer}

{"User guidance: " + user_instruction if user_instruction else ""}

Requirements:
1. Generate 1-3 tiers of progressive hints (from basic orientation to key insights)
2. Each hint should be concise and actionable (max {self.max_hint_chars} characters)
3. Hints should guide the student toward the solution without giving away the final answer
4. Focus on key geometric concepts, constructions, and reasoning steps
5. {"Translate hints to English" if self.force_english else "Preserve the original language"}
6. Style: {"Ask questions to guide thinking" if self.style == "socratic" else "Provide direct statements"}

Output format (JSON):
[
  {{
    "tier": 1,
    "label": "Initial orientation",
    "hints": [
      {{"text": "Hint text here", "from_steps": [0]}},
      {{"text": "Another hint", "from_steps": [0]}}
    ]
  }},
  {{
    "tier": 2,
    "label": "Key insight",
    "hints": [
      {{"text": "More specific hint", "from_steps": [0]}}
    ]
  }}
]

Generate hints that progressively reveal the solution approach without spoiling the final answer."""

            messages = [
                {"role": "system", "content": "You are a helpful geometry tutor assistant."},
                {"role": "user", "content": qa_prompt}
            ]
            
            # Call LLM to generate hints
            llm_result = self.call_llm_json(messages, expect="array")
            
            # Convert LLM result to HintTier objects
            hint_tiers = []
            if isinstance(llm_result, list):
                for i, tier_data in enumerate(llm_result[:self.max_tiers]):
                    if not isinstance(tier_data, dict):
                        continue
                    
                    hints = []
                    for hint_data in tier_data.get("hints", []):
                        if isinstance(hint_data, dict) and hint_data.get("text"):
                            hint = HintUnit(
                                text=hint_data.get("text", ""),
                                from_infer_indices=[0]  # All hints come from the Q&A, not inference steps
                            )
                            hints.append(hint)
                    
                    if hints:
                        tier = HintTier(
                            tier=i + 1,
                            label=tier_data.get("label", f"Tier {i + 1}"),
                            hints=hints
                        )
                        hint_tiers.append(tier)
            
            # Create hint sets
            hint_sets = []
            if hint_tiers:
                # Apply self-critique and fix
                fixed_tiers = self.self_critique_and_fix(hint_tiers, 1)  # Use 1 as infer_len since we have Q&A
                
                if fixed_tiers:
                    hint_set = HintSet(
                        difficulty=self.difficulty,
                        variant_id=1,
                        tiers=fixed_tiers
                    )
                    hint_sets.append(hint_set)
            
            # Extract technique tags (simplified for domain-agnostic use)
            technique_tags = ["helpful hints"]
            
            # Generate rationale
            total_hints = sum(len(tier.hints) for tier in hint_tiers)
            rationale = f"Generated {total_hints} hints directly from question and answer (inference process too short)."
            
            # Create result
            extraction_result = ExtractionResult(
                index=0,  # Will be updated by caller
                logic_tree=LogicTree(),  # Empty logic tree for Q&A generation
                hint_sets=hint_sets,
                technique_tags=technique_tags,
                rationale=rationale,
                echo_user_instruction=user_instruction,
                sanity_checks={
                    "math_markdown_ok": True,
                    "no_full_spoilers": True,
                    "indices_valid": True,
                    "spoiler_free": True,
                    "tiers_nonempty": len(hint_tiers) > 0,
                    "variants": len(hint_sets),
                    "logic_tree_valid": False,  # No logic tree for Q&A generation
                    "generated_from_qa": True  # Flag to indicate Q&A generation
                }
            )
            
            # Convert to dict for backward compatibility
            result_dict = extraction_result.model_dump()
            
            # Add legacy hint_tiers for backward compatibility
            if hint_tiers:
                legacy_tiers = []
                for tier in hint_tiers:
                    legacy_tier = {
                        "tier": tier.tier,
                        "label": tier.label,
                        "hints": [{"text": hint.text, "from_infer_indices": hint.from_infer_indices} 
                                 for hint in tier.hints]
                    }
                    legacy_tiers.append(legacy_tier)
                result_dict["hint_tiers"] = legacy_tiers
            else:
                result_dict["hint_tiers"] = []
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Failed to generate hints from Q&A: {e}")
            return self._create_empty_extraction_result(0, user_instruction)

    def generate_misleading_hints_from_qa(self, question: str, answer: str, num_versions: int = 4) -> List[Dict[str, str]]:
        """
        Generate misleading versions of a problem when inference process is too short.
        Uses question and answer to create subtly misleading hints.
        
        Args:
            question: The geometry problem question
            answer: The answer to the problem
            num_versions: Number of misleading versions to generate
            
        Returns:
            List of dictionaries with version number and misleading text
        """
        if not question or not answer:
            logger.warning("Empty question or answer for misleading hint generation")
            return [{"version": 0, "text": question}]
        
        versions = []
        # Version 0: Original problem
        versions.append({"version": 0, "text": question})
        
        try:
            # Generate misleading hints using Q&A
            misleading_prompt = f"""Generate {num_versions - 1} sets of MISLEADING hints for this academic problem. These hints should sound reasonable but deliberately lead students to WRONG answers.

Question: {question}
Correct Answer: {answer}

CRITICAL REQUIREMENTS FOR MISLEADING HINTS:
1. Each hint must suggest an INCORRECT approach, formula, or method
2. The hints should sound plausible and helpful but be fundamentally wrong
3. They should lead students away from the correct solution path
4. Make assumptions that seem reasonable but are actually false
5. Suggest using inappropriate theorems or methods
6. Include subtle mathematical errors that seem correct at first glance

IMPORTANT JSON FORMATTING RULES:
- Return ONLY valid JSON, no markdown or explanations
- Escape all backslashes in mathematical notation (use \\\\ instead of \\)
- Avoid control characters like tabs and newlines in strings
- Use simple quotes and avoid special characters when possible

Return ONLY a JSON array with this exact format:
[
  {{"version": 1, "hints": ["misleading hint 1", "misleading hint 2"]}},
  {{"version": 2, "hints": ["misleading hint 1", "misleading hint 2"]}}
]

Remember: These hints must be WRONG and misleading, not helpful!"""

            messages = [
                {"role": "system", "content": "You are an expert in creating educational misleading content for learning purposes."},
                {"role": "user", "content": misleading_prompt}
            ]
            
            # Call LLM to generate misleading hints
            llm_result = self.call_llm_json(messages, expect="array")
            
            if isinstance(llm_result, list) and llm_result:
                for item in llm_result:
                    if isinstance(item, dict) and "version" in item and "hints" in item:
                        version_num = item.get("version", len(versions))
                        hints = item.get("hints", [])
                        
                        if hints:
                            # Format hints as a single text block for the hint column
                            hint_text = "\n".join(f"- {h}" for h in hints if h)
                            
                            versions.append({
                                "version": version_num,
                                "text": question,  # Keep original question
                                "hint": hint_text  # Put hints in separate field
                            })
            else:
                logger.warning("LLM returned empty or invalid result for misleading hints generation")
            
            # Ensure we have at least the original version plus fallback versions
            if len(versions) == 1:
                logger.info("No valid misleading hints generated, creating fallback versions")
                # Fallback: create simple misleading versions based on common academic misconceptions
                fallback_hints = [
                    ["Try applying the most straightforward approach without considering edge cases"],
                    ["Assume the simplest case applies to all situations in this problem"],
                    ["Use the first formula that comes to mind without checking if it's appropriate"],
                    ["Focus on the most obvious interpretation without considering alternatives"]
                ]
                
                for i in range(1, min(num_versions, len(fallback_hints) + 1)):
                    hint_text = "\n".join(f"- {h}" for h in fallback_hints[i-1])
                    versions.append({
                        "version": i,
                        "text": question,  # Keep original question
                        "hint": hint_text  # Put hints in separate field
                    })
                
                # Fill remaining versions if needed
                for i in range(len(versions), num_versions):
                    versions.append({"version": i, "text": question, "hint": ""})
                    
        except Exception as e:
            logger.error(f"Failed to generate misleading hints from Q&A: {e}")
            # Return just original question for all versions on error
            for i in range(1, num_versions):
                versions.append({"version": i, "text": question, "hint": ""})
        
        return versions

    def generate_helpful_hints_from_qa(self, question: str, answer: str, num_versions: int = 4) -> List[Dict[str, str]]:
        """
        Generate helpful versions of a problem when inference process is too short.
        Uses question and answer to create progressive helpful hints.
        
        Args:
            question: The academic problem question
            answer: The answer to the problem
            num_versions: Number of helpful versions to generate
            
        Returns:
            List of dictionaries with version number and helpful hint text
        """
        if not question or not answer:
            logger.warning("Empty question or answer for helpful hint generation")
            return [{"version": 0, "text": question, "hint": ""}]
        
        versions = []
        # Version 0: Original problem
        versions.append({"version": 0, "text": question, "hint": ""})
        
        try:
            # Generate helpful hints using Q&A
            helpful_prompt = f"""Generate {num_versions - 1} sets of helpful hints for this academic problem. Each hint should guide students toward the correct solution.

Question: {question}
Correct Answer: {answer}

IMPORTANT JSON FORMATTING RULES:
- Return ONLY valid JSON, no markdown or explanations
- Escape all backslashes in mathematical notation (use \\\\ instead of \\)
- Avoid control characters like tabs and newlines in strings
- Use simple quotes and avoid special characters when possible

Return ONLY a JSON array with this exact format:
[
  {{"version": 1, "hints": ["helpful hint 1", "helpful hint 2"]}},
  {{"version": 2, "hints": ["more specific hint 1", "more specific hint 2"]}}
]

Make hints progressively more specific and helpful."""

            messages = [
                {"role": "system", "content": "You are an expert academic tutor creating helpful educational content."},
                {"role": "user", "content": helpful_prompt}
            ]
            
            # Call LLM to generate helpful hints
            llm_result = self.call_llm_json(messages, expect="array")
            
            if isinstance(llm_result, list) and llm_result:
                for item in llm_result:
                    if isinstance(item, dict) and "version" in item and "hints" in item:
                        version_num = item.get("version", len(versions))
                        hints = item.get("hints", [])
                        
                        if hints:
                            # Format hints as a single text block for the hint column
                            hint_text = "\n".join(f"- {h}" for h in hints if h)
                            
                            versions.append({
                                "version": version_num,
                                "text": question,  # Keep original question
                                "hint": hint_text  # Put hints in separate field
                            })
            else:
                logger.warning("LLM returned empty or invalid result for helpful hints generation")
            
            # Ensure we have at least the original version plus fallback versions
            if len(versions) == 1:
                logger.info("No valid helpful hints generated, creating fallback versions")
                # Fallback: create simple helpful versions based on common academic approaches
                fallback_hints = [
                    ["Start by identifying what information is given and what you need to find"],
                    ["Look for key relationships or patterns in the given data"],
                    ["Consider which formulas or methods are most relevant to this type of problem"],
                    ["Break the problem into smaller, manageable steps"]
                ]
                
                for i in range(1, min(num_versions, len(fallback_hints) + 1)):
                    hint_text = "\n".join(f"- {h}" for h in fallback_hints[i-1])
                    versions.append({
                        "version": i,
                        "text": question,  # Keep original question
                        "hint": hint_text  # Put hints in separate field
                    })
                
                # Fill remaining versions if needed
                for i in range(len(versions), num_versions):
                    versions.append({"version": i, "text": question, "hint": ""})
                    
        except Exception as e:
            logger.error(f"Failed to generate helpful hints from Q&A: {e}")
            # Return just original question for all versions on error
            for i in range(1, num_versions):
                versions.append({"version": i, "text": question, "hint": ""})
        
        return versions

    def _extract_tree_hints_with_size_control(self, record: dict, target_size: int) -> List[str]:
        """
        Extract hints using logic tree strategy with expand_size adaptation.
        
        Args:
            record: JSONL record with infer_process
            target_size: Required number of hint variants
            
        Returns:
            List of hint strings with exactly target_size elements
        """
        # First extract hints using traditional logic tree approach
        extraction_result = self.extract_hints(record)
        
        # Build hint progression from extraction result
        hint_sets = extraction_result.get('hint_sets', [])
        if not hint_sets:
            # Try legacy hint_tiers format
            legacy_tiers = extraction_result.get('hint_tiers', [])
            if legacy_tiers:
                hint_progression = self.build_hint_progression_from_legacy(legacy_tiers)
            else:
                hint_progression = []
        else:
            hint_progression = self.build_hint_progression(hint_sets)
        
        if not hint_progression:
            # No hints extracted, return empty list (will be handled by fallback)
            return []
        
        # Adapt hints to target size
        if len(hint_progression) > target_size:
            # Tree is too deep: group hints progressively
            logger.debug(f"Tree too deep ({len(hint_progression)} hints), grouping into {target_size} progressive groups")
            return self._group_hints_progressively(hint_progression, target_size)
        
        elif len(hint_progression) < target_size:
            # Tree is too shallow: generate additional hints
            logger.debug(f"Tree too shallow ({len(hint_progression)} hints), generating {target_size - len(hint_progression)} additional hints")
            return self._extend_hints_with_llm(record, hint_progression, target_size)
        
        else:
            # Perfect match: return as is
            return hint_progression

    def _group_hints_progressively(self, hint_progression: List[str], target_size: int) -> List[str]:
        """
        Group hints progressively when there are too many hints for the target size.
        
        Example: hints [1,2,3,4,5,6,7,8] with target_size=2
        Result: ["1,2,3,4", "1,2,3,4,5,6,7,8"] (progressive accumulation)
        """
        if not hint_progression or target_size <= 0:
            return []
        
        grouped_hints = []
        total_hints = len(hint_progression)
        
        for i in range(target_size):
            # Calculate how many hints to include in this group
            # Distribute hints progressively: earlier groups get fewer hints
            hints_in_group = int((i + 1) * total_hints / target_size)
            
            # Take the first N hints and combine them
            current_hints = hint_progression[:hints_in_group]
            
            if len(current_hints) == 1:
                # Single hint: use as is
                grouped_hints.append(current_hints[0])
            else:
                # Multiple hints: combine them progressively
                combined_hint = "\n".join(f"- {hint.strip()}" for hint in current_hints if hint.strip())
                grouped_hints.append(combined_hint)
        
        return grouped_hints

    def _extend_hints_with_llm(self, record: dict, existing_hints: List[str], target_size: int) -> List[str]:
        """
        Generate additional hints when there are too few hints for the target size.
        """
        if target_size <= len(existing_hints):
            return existing_hints[:target_size]
        
        # Get question and answer for context
        question = record.get('question', '')
        infer_process = record.get('infer_process', [])
        
        # Create context from existing hints and inference process
        context_parts = []
        if existing_hints:
            context_parts.append("Existing hints:\n" + "\n".join(f"- {h}" for h in existing_hints))
        if infer_process:
            context_parts.append("Solution steps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(infer_process)))
        
        context = "\n\n".join(context_parts)
        missing_count = target_size - len(existing_hints)
        
        try:
            # Generate additional hints using LLM
            extend_prompt = f"""Based on the following academic problem and context, generate {missing_count} additional helpful hints that complement the existing ones.

Question: {question}

{context}

Generate {missing_count} new helpful hints that:
1. Build upon or complement the existing hints
2. Provide different perspectives or approaches
3. Are progressively more specific
4. Help students solve the problem step by step

IMPORTANT JSON FORMATTING RULES:
- Return ONLY valid JSON, no markdown or explanations
- Escape all backslashes in mathematical notation (use \\\\ instead of \\)
- Avoid control characters like tabs and newlines in strings
- Use simple quotes and avoid special characters when possible

Return ONLY a JSON array of hint strings:
["additional hint 1", "additional hint 2", ...]"""

            messages = [
                {"role": "system", "content": "You are an expert academic tutor creating progressive educational hints."},
                {"role": "user", "content": extend_prompt}
            ]
            
            # Call LLM to generate additional hints
            llm_result = self.call_llm_json(messages, expect="array")
            
            additional_hints = []
            if isinstance(llm_result, list):
                for hint in llm_result[:missing_count]:  # Take only what we need
                    if isinstance(hint, str) and hint.strip():
                        additional_hints.append(hint.strip())
            
            # Combine existing and new hints
            all_hints = existing_hints + additional_hints
            
            # If still not enough, pad with generic hints
            while len(all_hints) < target_size:
                all_hints.append("Consider the relationships and patterns in the given information")
            
            return all_hints[:target_size]  # Ensure exact size
            
        except Exception as e:
            logger.warning(f"Failed to generate additional hints: {e}")
            # Fallback: repeat existing hints to reach target size
            extended_hints = existing_hints.copy()
            while len(extended_hints) < target_size:
                extended_hints.extend(existing_hints[:target_size - len(extended_hints)])
            return extended_hints[:target_size]

    def extract_hints(self, record: dict) -> dict:
        """
        Extract academic hints from a single JSONL record using the new pipeline.
        
        Args:
            record: A dictionary containing infer_process and user_instruction
            
        Returns:
            A dictionary representation of ExtractionResult
        """
        index = record.get("index", 0)
        infer_process = record.get("infer_process", [])
        user_instruction = record.get("user_instruction", "")
        
        if not infer_process:
            logger.warning(f"Empty infer_process for record {index}")
            return self._create_empty_extraction_result(index, user_instruction)
        
        # Check if inference process is too short
        if len(infer_process) < self.SHORT_INFERENCE_THRESHOLD:
            logger.info(f"Inference process too short ({len(infer_process)} steps < {self.SHORT_INFERENCE_THRESHOLD}) for record {index}, using Q&A-based hint generation")
            
            # Try to get question and answer from record
            question = record.get("question", "")
            answer = record.get("answer", "")
            
            if question and answer:
                # Use Q&A-based hint generation
                result = self.generate_hints_from_qa(question, answer, user_instruction)
                result["index"] = index  # Update with correct index
                return result
            else:
                logger.warning(f"No question or answer available for Q&A-based hint generation for record {index}")
                return self._create_empty_extraction_result(index, user_instruction)
        
        try:
            if self.num_workers > 1:
                # Parallel processing: submit logic tree and hint tier planning concurrently
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Step 1: Build logic tree (concurrent with other tasks)
                    logic_tree_future = executor.submit(self.build_logic_tree, infer_process)
                    
                    # Wait for logic tree to complete before planning tiers
                    logic_tree = logic_tree_future.result()
                    
                    # Step 2: Plan hint tiers 
                    base_tiers = self.plan_hint_tiers(logic_tree, self.difficulty, self.max_tiers, self.style)
                    
                    # Step 3: Generate variants (skip for efficiency in TSV mode)
                    variant_tier_lists = [base_tiers]
            else:
                # Sequential processing (original logic)
                # Step 1: Build logic tree
                logic_tree = self.build_logic_tree(infer_process)
                
                # Step 2: Plan hint tiers
                base_tiers = self.plan_hint_tiers(logic_tree, self.difficulty, self.max_tiers, self.style)
                
                # Step 3: Generate variants (skip for efficiency)
                variant_tier_lists = [base_tiers]
            
            # Step 4: Self-critique and fix each variant
            hint_sets = []
            if self.num_workers > 1 and len(variant_tier_lists) > 1:
                # Parallel self-critique for multiple variants
                with ThreadPoolExecutor(max_workers=min(self.num_workers, len(variant_tier_lists))) as executor:
                    # Submit all variant processing tasks
                    future_to_variant = {}
                    for variant_id, variant_tiers in enumerate(variant_tier_lists):
                        future = executor.submit(self.self_critique_and_fix, variant_tiers, len(infer_process))
                        future_to_variant[future] = variant_id
                    
                    # Collect results
                    for future in future_to_variant:
                        variant_id = future_to_variant[future]
                        try:
                            fixed_tiers = future.result()
                            if fixed_tiers:
                                hint_set = HintSet(
                                    difficulty=self.difficulty,
                                    variant_id=variant_id + 1,
                                    tiers=fixed_tiers
                                )
                                hint_sets.append(hint_set)
                        except Exception as e:
                            logger.warning(f"Failed to process variant {variant_id}: {e}")
            else:
                # Sequential processing for single variant or when num_workers=1
                for variant_id, variant_tiers in enumerate(variant_tier_lists):
                    fixed_tiers = self.self_critique_and_fix(variant_tiers, len(infer_process))
                    
                    if fixed_tiers:
                        hint_set = HintSet(
                            difficulty=self.difficulty,
                            variant_id=variant_id + 1,
                            tiers=fixed_tiers
                        )
                        hint_sets.append(hint_set)
            
            # Step 5: Extract technique tags from all hint sets
            all_tiers = []
            for hint_set in hint_sets:
                all_tiers.extend(hint_set.tiers)
            technique_tags = ["helpful hints"]  # Simplified for domain-agnostic use
            
            # Step 6: Generate rationale
            total_hints = sum(len(tier.hints) for tier in all_tiers)
            if len(hint_sets) > 1:
                rationale = f"Organized {total_hints} hints into {len(base_tiers)} tiers across {len(hint_sets)} variants at '{self.difficulty}' difficulty level."
            else:
                rationale = f"Organized {total_hints} hints into {len(base_tiers)} tiers at '{self.difficulty}' difficulty level."
            
            # Step 7: Create result with sanity checks
            sanity_checks = {
                "math_markdown_ok": True,
                "no_full_spoilers": True,
                "indices_valid": True,
                "spoiler_free": True,
                "tiers_nonempty": len(all_tiers) > 0,
                "variants": len(hint_sets),
                "logic_tree_valid": len(logic_tree.nodes) > 0
            }
            
            # Create ExtractionResult
            extraction_result = ExtractionResult(
                index=index,
                logic_tree=logic_tree,
                hint_sets=hint_sets,
                technique_tags=technique_tags,
                rationale=rationale,
                echo_user_instruction=user_instruction,
                sanity_checks=sanity_checks
            )
            
            # Convert to dict for backward compatibility
            result_dict = extraction_result.model_dump()
            
            # Add legacy hint_tiers for backward compatibility (use standard difficulty, variant 1)
            if hint_sets:
                standard_set = None
                for hint_set in hint_sets:
                    if hint_set.difficulty == "standard" or (not standard_set and hint_set.variant_id == 1):
                        standard_set = hint_set
                        break
                
                if not standard_set:
                    standard_set = hint_sets[0]
                
                # Convert to legacy format
                legacy_tiers = []
                for tier in standard_set.tiers:
                    legacy_tier = {
                        "tier": tier.tier,
                        "label": tier.label,
                        "hints": [{"text": hint.text, "from_infer_indices": hint.from_infer_indices} 
                                 for hint in tier.hints]
                    }
                    legacy_tiers.append(legacy_tier)
                
                result_dict["hint_tiers"] = legacy_tiers
            else:
                result_dict["hint_tiers"] = []
            
            # Update metrics
            self.processed_count += 1
            self.emitted_tiers_total += len(all_tiers)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Pipeline failed for record {index}: {e}")
            self.failed_count += 1
            return self._create_empty_extraction_result(index, user_instruction)

    def _create_empty_extraction_result(self, index: int, user_instruction: str) -> dict:
        """Create an empty ExtractionResult for failed extractions."""
        empty_result = ExtractionResult(
            index=index,
            logic_tree=LogicTree(),
            hint_sets=[],
            technique_tags=[],
            rationale="Failed to extract hints.",
            echo_user_instruction=user_instruction,
            sanity_checks={
                "math_markdown_ok": False,
                "no_full_spoilers": True,
                "indices_valid": True,
                "spoiler_free": True,
                "tiers_nonempty": False,
                "variants": 0,
                "logic_tree_valid": False
            }
        )
        
        result_dict = empty_result.model_dump()
        result_dict["hint_tiers"] = []  # Add legacy field
        return result_dict
    # Illustration:
    # record = {"index": 159, "infer_process": ["Let the curve intersect the sphere at points \\(A\\) and \\(B\\).", 
    #                                            "Let \\(B'\\) be the point diametrically opposite \\(A\\)."], 
    #           "user_instruction": ""}
    # result = extractor.extract_hints(record)
    # 
    # Result might be:
    # {
    #   "index": 159,
    #   "hint_tiers": [
    #     {
    #       "tier": 1,
    #       "label": "Key Construction",
    #       "hints": [{"text": "Consider the point diametrically opposite to A.", "from_infer_indices": [1]}]
    #     }
    #   ],
    #   "technique_tags": ["diametric points"],
    #   "rationale": "The key insight is introducing the diametrically opposite point.",
    #   "echo_user_instruction": "",
    #   "sanity_checks": {"math_markdown_ok": true, "no_full_spoilers": true, "indices_valid": true}
    # }

    def clamp_and_validate(self, result: dict, infer_len: int) -> dict:
        """
        Validate and fix issues with the LLM result.
        
        Args:
            result: The LLM result dictionary
            infer_len: The length of the original infer_process array
            
        Returns:
            A validated and cleaned result dictionary
        """
        # Initialize or default missing fields
        result.setdefault("hint_tiers", [])
        result.setdefault("technique_tags", [])
        result.setdefault("rationale", "")
        result.setdefault("echo_user_instruction", "")
        result.setdefault("sanity_checks", {
            "math_markdown_ok": True,
            "no_full_spoilers": True,
            "indices_valid": True
        })
        
        # Iterate through tiers and hints to validate and trim
        valid_tiers = []
        for tier_data in result["hint_tiers"]:
            if not isinstance(tier_data, dict):
                continue
                
            tier_data.setdefault("tier", len(valid_tiers) + 1)
            tier_data.setdefault("label", f"Hint Tier {tier_data['tier']}")
            tier_data.setdefault("hints", [])
            
            valid_hints = []
            for hint in tier_data["hints"]:
                if not isinstance(hint, dict):
                    continue
                    
                # Validate hint text exists and trim if necessary
                if "text" not in hint or not isinstance(hint["text"], str) or not hint["text"].strip():
                    continue
                    
                hint["text"] = hint["text"].strip()
                if len(hint["text"]) > self.max_hint_chars:
                    # Trim while preserving meaning
                    hint["text"] = hint["text"][:self.max_hint_chars-3] + "..."
                
                # Validate indices
                if "from_infer_indices" not in hint or not isinstance(hint["from_infer_indices"], list):
                    hint["from_infer_indices"] = [0]  # Default to first step
                
                # Fix invalid indices
                hint["from_infer_indices"] = [
                    i for i in hint["from_infer_indices"] 
                    if isinstance(i, int) and 0 <= i < infer_len
                ]
                
                # Skip if no valid indices
                if not hint["from_infer_indices"]:
                    hint["from_infer_indices"] = [0]  # Default to first step
                
                valid_hints.append(hint)
            
            if valid_hints:
                tier_data["hints"] = valid_hints
                valid_tiers.append(tier_data)
        
        result["hint_tiers"] = valid_tiers
        
        # Update sanity checks based on validation
        result["sanity_checks"]["indices_valid"] = all(
            all(0 <= idx < infer_len for idx in hint.get("from_infer_indices", []))
            for tier in result["hint_tiers"]
            for hint in tier.get("hints", [])
        )
        
        # Check for math markdown integrity (very basic check)
        has_math = any("\\" in hint["text"] for tier in result["hint_tiers"] for hint in tier["hints"])
        if has_math:
            unbalanced_parens = any(
                hint["text"].count("\\(") != hint["text"].count("\\)") 
                for tier in result["hint_tiers"] 
                for hint in tier["hints"]
            )
            result["sanity_checks"]["math_markdown_ok"] = not unbalanced_parens
        
        return result

    # Illustration:
    # result = {
    #   "hint_tiers": [
    #     {"tier": 1, "label": "Key Insight", "hints": [
    #       {"text": "A very long hint that exceeds the maximum character limit of 220...", 
    #        "from_infer_indices": [0, 100]}
    #     ]}
    #   ]
    # }
    # validated = extractor.clamp_and_validate(result, 10)
    # 
    # The hint text is trimmed to 220 chars, and the invalid index 100 is removed,
    # leaving only [0]. Sanity checks are updated accordingly.

    def process_stream(self, in_path: str, out_path: str, dry_n: Optional[int] = None) -> None:
        """
        Process a JSONL file of geometry solutions and extract hints.
        
        Args:
            in_path: Path to the input JSONL file
            out_path: Path to the output JSON file (with proper indentation)
            dry_n: If set, process only the first N record
        """
        # Reset metrics
        self.processed_count = 0
        self.failed_count = 0
        self.emitted_tiers_total = 0
        
        # Count lines for progress bar
        line_count = 0
        with open(in_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1
        
        # All results will be stored in this list
        all_results = []
        
        # Open files for reading
        with open(in_path, 'r', encoding='utf-8') as infile:
            try:
                # Setup progress bar
                with tqdm(total=min(line_count, dry_n or line_count)) as pbar:
                    for i, line in enumerate(infile):
                        if dry_n is not None and i >= dry_n:
                            break
                            
                        try:
                            # Parse input
                            record = json.loads(line.strip())
                            
                            # Extract hints
                            result = self.extract_hints(record)
                            
                            # Print to stdout for dry run, otherwise add to results list
                            all_results.append(result)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed JSON at line {i+1}")
                        except Exception as e:
                            logger.error(f"Error processing line {i+1}: {e}")
                        
                        pbar.update(1)
                
                # Write all results to the output file
                if out_path:
                    with open(out_path, 'w', encoding='utf-8') as outfile:
                        json.dump(all_results, outfile, indent=4, ensure_ascii=False)
                        logger.info(f"Wrote {len(all_results)} records to {out_path}")
                
                # Log metrics
                logger.info(f"Processed {self.processed_count} records")
                logger.info(f"Failed LLM calls: {self.failed_count}")
                
                if self.processed_count > 0:
                    avg_tiers = self.emitted_tiers_total / self.processed_count
                    logger.info(f"Average tiers per record: {avg_tiers:.2f}")
                    
            except Exception as e:
                logger.error(f"Error processing file: {e}")

    # Illustration:
    # extractor.process_stream(
    #     in_path="/data/geometry_problems.jsonl", 
    #     out_path="/data/geometry_problems.hints.jsonl",
    #     dry_n=None  # Process the entire file
    # )
    #
    # The function reads each line from the input file, extracts hints using the LLM,
    # validates the result, and writes it to the output file. For malformed lines,
    # it logs a warning and continues processing.

    def process_stream_concurrent(self, in_path: str, out_path: str, dry_n: Optional[int] = None, num_workers: Optional[int] = None) -> None:
        """
        Process a JSONL file with concurrent hint extraction.
        
        Args:
            in_path: Path to the input JSONL file
            out_path: Path to the output JSON file (with proper indentation)
            dry_n: If set, process only the first N records
            num_workers: Number of worker threads for parallel processing
        """
        # Use instance variable if num_workers not provided
        if num_workers is None:
            num_workers = self.num_workers
            
        # Reset metrics
        self.processed_count = 0
        self.failed_count = 0
        self.emitted_tiers_total = 0
        
        # Read all records first
        records = []
        with open(in_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                if dry_n is not None and i >= dry_n:
                    break
                    
                try:
                    record = json.loads(line.strip())
                    records.append((i, record))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {i+1}")
        
        # Process records concurrently
        all_results = [None] * len(records)  # Pre-allocate to maintain order
        
        def process_record(index_record):
            i, record = index_record
            try:
                result = self.extract_hints(record)
                return i, result
            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                return i, self._create_empty_extraction_result(record.get("index", i), record.get("user_instruction", ""))
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_record, record_data): record_data[0] 
                             for record_data in records}
            
            # Collect results with progress bar
            with tqdm(total=len(records)) as pbar:
                for future in as_completed(future_to_index):
                    try:
                        i, result = future.result()
                        all_results[i] = result
                        pbar.update(1)
                    except Exception as e:
                        i = future_to_index[future]
                        logger.error(f"Failed to process record {i}: {e}")
                        pbar.update(1)
        
        # Filter out None results and write to file
        valid_results = [result for result in all_results if result is not None]
        
        if out_path:
            with open(out_path, 'w', encoding='utf-8') as outfile:
                json.dump(valid_results, outfile, indent=4, ensure_ascii=False)
                logger.info(f"Wrote {len(valid_results)} records to {out_path}")
        
        # Log metrics
        logger.info(f"Processed {self.processed_count} records")
        logger.info(f"Failed LLM calls: {self.failed_count}")
        
        if self.processed_count > 0:
            avg_tiers = self.emitted_tiers_total / self.processed_count
            logger.info(f"Average tiers per record: {avg_tiers:.2f}")

    def build_hint_progression(self, hint_sets: List[HintSet]) -> List[str]:
        """
        Build progressive hint list using hint-level granularity.
        Returns individual hints, not cumulative ones.
        
        Args:
            hint_sets: List of hint sets from extraction
            
        Returns:
            List of individual hint strings (not cumulative)
        """
        if not hint_sets:
            return []
        
        # Use the first hint set (or find standard difficulty if available)
        chosen_set = hint_sets[0]
        for hint_set in hint_sets:
            # Handle both dict and object formats
            difficulty = hint_set.get('difficulty') if isinstance(hint_set, dict) else getattr(hint_set, 'difficulty', None)
            if difficulty == "standard":
                chosen_set = hint_set
                break
        
        # Flatten all hints in tier order (using hint-level granularity)
        all_hints = []
        # Handle both dict and object formats for tiers
        tiers = chosen_set.get('tiers') if isinstance(chosen_set, dict) else getattr(chosen_set, 'tiers', [])
        for tier in sorted(tiers, key=lambda t: t.get('tier') if isinstance(t, dict) else getattr(t, 'tier', 0)):
            # Handle both dict and object formats for hints
            hints = tier.get('hints') if isinstance(tier, dict) else getattr(tier, 'hints', [])
            for hint in hints:
                hint_text = hint.get('text') if isinstance(hint, dict) else getattr(hint, 'text', '')
                all_hints.append(hint_text.strip())
        
        # Build individual hint progression: each level contains just one hint
        progression = []
        for hint in all_hints:
            progression.append(f"- {hint}")
        
        return progression

    def build_hint_progression_from_legacy(self, legacy_tiers: List[Dict], granularity: str) -> List[str]:
        """
        Build progressive hint list from legacy hint_tiers format.
        
        Args:
            legacy_tiers: List of legacy tier dictionaries
            granularity: "hint" or "tier"
            
        Returns:
            List of individual hint strings (not cumulative)
        """
        if not legacy_tiers:
            return []
        
        if granularity == "hint":
            # Flatten all hints from legacy tiers
            all_hints = []
            for tier in sorted(legacy_tiers, key=lambda t: t.get('tier', 0)):
                hints = tier.get('hints', [])
                for hint in hints:
                    hint_text = hint.get('text', '') if isinstance(hint, dict) else str(hint)
                    all_hints.append(hint_text.strip())
            
            # Build individual hint progression
            progression = []
            for hint in all_hints:
                progression.append(f"- {hint}")
            
            return progression
            
        elif granularity == "tier":
            # Build individual tier progression from legacy format
            sorted_tiers = sorted(legacy_tiers, key=lambda t: t.get('tier', 0))
            progression = []
            
            for tier in sorted_tiers:
                tier_hints = []
                hints = tier.get('hints', [])
                for hint in hints:
                    hint_text = hint.get('text', '') if isinstance(hint, dict) else str(hint)
                    tier_hints.append(hint_text.strip())
                
                if tier_hints:
                    tier_combined = "\n".join([f"- {h}" for h in tier_hints])
                    progression.append(tier_combined)
            
            return progression
        
        else:
            logger.warning(f"Unknown granularity: {granularity}")
            return []

    def expand_tsv_row(self, original_row: Dict[str, str], hint_progression: List[str], 
                      granularity: str, start_index: int = 1) -> List[Dict[str, str]]:
        """
        Expand a single TSV row into multiple variants with progressive hints.
        
        Args:
            original_row: Original TSV row data
            hint_progression: List of cumulative hint strings (already progressive)
            granularity: "hint" or "tier" 
            start_index: Starting index for the variants
            
        Returns:
            List of expanded row dictionaries with progressive hints
        """
        expanded_rows = []
        
        # Calculate how many variants to create (no original row, only hint variants)
        max_variants = len(hint_progression) if hint_progression else 1
        
        for i in range(max_variants):
            # Create new row by copying original (preserves question, answer, etc.)
            new_row = original_row.copy()
            
            # Assign sequential unique index
            new_row['index'] = str(start_index + i)
            
            # Update hint field progressively
            if i < len(hint_progression):
                new_row['hint'] = hint_progression[i]
            else:
                # Fallback: use last available hint or empty
                new_row['hint'] = hint_progression[-1] if hint_progression else ""
            
            expanded_rows.append(new_row)
        
        return expanded_rows

    def process_tsv_expansion(self, jsonl_path: str, tsv_in_path: str, tsv_out_path: str, 
                             dry_n: Optional[int] = None) -> None:
        """
        Process TSV expansion by combining JSONL hint extraction with TSV row expansion.
        
        Args:
            jsonl_path: Path to input JSONL file with infer_process
            tsv_in_path: Path to input TSV file
            tsv_out_path: Path to output expanded TSV file
            dry_n: If set, process only the first N JSONL records
        """
        # Parse evolvement parameter
        evolvement_types = [e.strip().lower() for e in self.evolvement.split(',')]
        logger.info(f"Evolvement types: {evolvement_types}")
        
        # Read TSV data
        logger.info(f"Reading TSV data from {tsv_in_path}")
        tsv_data = read_tsv(tsv_in_path)
        
        if not tsv_data:
            logger.error("No valid TSV data found")
            return
        
        # Get column order from first row
        first_row = next(iter(tsv_data.values()))
        fieldnames = list(first_row.keys())
        
        # Process JSONL and expand TSV rows
        expanded_rows = []
        processed_indices = set()
        next_index = 1  # Start assigning indices from 1
        
        # Count lines for progress bar
        line_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1
        
        # Apply dry-run limit if specified
        if dry_n is not None:
            line_count = min(line_count, dry_n)
            logger.info(f"Dry run mode: processing first {line_count} JSONL records")
        else:
            logger.info(f"Processing {line_count} JSONL records for hint extraction")
        
        # Read all records first for parallel processing
        all_records = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Apply dry-run limit
                if dry_n is not None and line_num > dry_n:
                    break
                try:
                    record = json.loads(line.strip())
                    record_index = str(record.get('index', ''))
                    
                    if not record_index:
                        logger.warning(f"Skipping JSONL record at line {line_num}: no index")
                        continue
                    
                    # Find corresponding TSV row
                    if record_index not in tsv_data:
                        logger.warning(f"No TSV row found for index {record_index}")
                        continue
                    
                    tsv_row = tsv_data[record_index]
                    all_records.append((record, tsv_row, record_index))
                    
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_num}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
        
        # Process records (parallel or sequential based on num_workers)
        if self.num_workers > 1 and len(all_records) > 1:
            logger.info(f"Using {self.num_workers} workers for parallel TSV expansion")
            # Parallel processing
            self._process_tsv_records_parallel(all_records, expanded_rows, processed_indices, 
                                             evolvement_types, granularity, next_index)
        else:
            # Sequential processing (original logic)
            self._process_tsv_records_sequential(all_records, expanded_rows, processed_indices, 
                                               evolvement_types, granularity, next_index)
        
        # Note: We no longer add unprocessed TSV rows since we only want expanded variants
        
        # Sort rows by index (now pure numbers) for stable output and reassign sequential indices
        def sort_key(row):
            try:
                return int(row['index'])
            except (ValueError, TypeError):
                return float('inf')  # Put invalid indices at the end
        
        expanded_rows.sort(key=sort_key)
        
        # Reassign sequential indices starting from 1
        for i, row in enumerate(expanded_rows):
            row['index'] = str(i + 1)
        
        # Write expanded TSV
        logger.info(f"Writing {len(expanded_rows)} expanded rows to {tsv_out_path}")
        write_tsv(tsv_out_path, expanded_rows, fieldnames)
        
        # Log summary statistics
        original_count = len(tsv_data)
        expanded_count = len(expanded_rows)
        avg_variants = expanded_count / original_count if original_count > 0 else 0
        
        logger.info(f"TSV expansion complete:")
        logger.info(f"  Input rows: {original_count}")
        logger.info(f"  Output rows: {expanded_count}")
        logger.info(f"  Average variants per question: {avg_variants:.2f}")
        logger.info(f"  Processed indices: {len(processed_indices)}")

    def _process_single_tsv_record(self, record_data: Tuple, evolvement_types: List[str], 
                                  granularity: str, start_index: int) -> Tuple[List[Dict], str]:
        """
        Process a single TSV record and return expanded variants with precise size control.
        
        Args:
            record_data: Tuple of (record, tsv_row, record_index)
            evolvement_types: List of evolvement types to process
            granularity: Expansion granularity  
            start_index: Starting index for variants
            
        Returns:
            Tuple of (expanded_variants, record_index)
        """
        record, tsv_row, record_index = record_data
        
        # Get question and answer from TSV row
        question = tsv_row.get('question', '')
        answer = tsv_row.get('answer', '')
        
        # Validate answer consistency if present
        if 'answer' in record and record['answer']:
            if answer and str(record['answer']).strip() != str(answer).strip():
                logger.warning(f"Answer mismatch for index {record_index}: "
                             f"JSONL='{record['answer']}' vs TSV='{answer}'")
        
        # Generate expansions based on evolvement types
        logger.debug(f"Processing evolvement types for index {record_index}: {evolvement_types}")
        all_hint_versions = []
        
        # Check if inference process is too short for both easy and hard modes
        infer_process = record.get("infer_process", [])
        use_qa_approach = len(infer_process) < self.SHORT_INFERENCE_THRESHOLD
        
        for evo_type in evolvement_types:
            if evo_type == "easy":
                if use_qa_approach and question and answer:
                    # Use QA-based helpful hint generation for short inference processes
                    # logger.info(f"Using QA-based helpful hint generation for index {record_index}")
                    helpful_versions = self.generate_helpful_hints_from_qa(question, answer, self.expand_size)
                    # Skip version 0 (original), only take the hint versions
                    for version in helpful_versions[1:]:
                        all_hint_versions.append(version.get('hint', ''))
                else:
                    # Use traditional logic tree hint extraction with expand_size adaptation
                    logger.debug(f"Extracting helpful hints using logic tree for index {record_index}")
                    # Try Q&A approach first even for longer inference processes to reduce API calls
                    if question and answer:
                        logger.debug(f"Using Q&A approach for efficiency for index {record_index}")
                        helpful_versions = self.generate_helpful_hints_from_qa(question, answer, self.expand_size)
                        # Skip version 0 (original), only take the hint versions
                        for version in helpful_versions[1:]:
                            all_hint_versions.append(version.get('hint', ''))
                    else:
                        # Fallback to expensive tree extraction only if no Q&A available
                        tree_hints = self._extract_tree_hints_with_size_control(record, granularity, self.expand_size)
                        all_hint_versions.extend(tree_hints)
            
            elif evo_type == "hard":
                if use_qa_approach and question and answer:
                    # Use QA-based misleading hint generation
                    # logger.info(f"Using QA-based misleading hint generation for index {record_index}")
                    misleading_versions = self.generate_misleading_hints_from_qa(question, answer, self.expand_size)
                    # Skip version 0 (original), only take the hint versions
                    for version in misleading_versions[1:]:
                        all_hint_versions.append(version.get('hint', ''))
                else:
                    # Use traditional misleading hint generation
                    logger.debug(f"Generating misleading hints for index {record_index}")
                    # Try Q&A approach first even for longer inference processes to reduce API calls
                    if question and answer:
                        logger.debug(f"Using Q&A approach for efficiency for index {record_index}")
                        misleading_versions = self.generate_misleading_hints_from_qa(question, answer, self.expand_size)
                        # Skip version 0 (original), only take the hint versions
                        for version in misleading_versions[1:]:
                            all_hint_versions.append(version.get('hint', ''))
                    else:
                        # Fallback to expensive inference-based generation only if no Q&A available
                        try:
                            # Add question and answer to record if not present
                            if question and 'question' not in record:
                                record['question'] = question
                            if answer and 'answer' not in record:
                                record['answer'] = answer
                            
                            misleading_versions = self.generate_misleading_versions(record, num_versions=self.expand_size)
                            
                            # Extract hints from versions
                            for version in misleading_versions[1:]:  # Skip version 0 (original)
                                hint = version.get('hint', '')
                                if not hint and 'text' in version:
                                    # Fallback: extract hint from text if needed
                                    version_text = version.get('text', '')
                                    if '\n\n' in version_text:
                                        hint = version_text.split('\n\n', 1)[1]
                                all_hint_versions.append(hint)
                        
                        except Exception as e:
                            logger.warning(f"Failed to generate misleading variants for index {record_index}: {e}")
        
        # Apply precise size control
        target_size = self.expand_size
        
        # If we have no hints, create fallback hints
        if not all_hint_versions:
            logger.info(f"No hints generated for index {record_index}, creating fallback hints")
            if 'hard' in evolvement_types:
                fallback_hints = [
                    "Try applying the most obvious approach first",
                    "Consider that similar-looking problems have the same solutions",
                    "Remember that approximations are often exact in academic problems",
                    "Focus on the most straightforward interpretation"
                ]
            else:
                fallback_hints = [
                    "Break down the problem into smaller parts",
                    "Consider what principles from the subject apply here", 
                    "Think about the relationships between the given information",
                    "Start by identifying what information is given and what you need to find"
                ]
            all_hint_versions = fallback_hints[:target_size]
        
        # Ensure exact size: truncate if too long, repeat if too short
        while len(all_hint_versions) < target_size:
            # Repeat the hints if we don't have enough
            if all_hint_versions:
                all_hint_versions.extend(all_hint_versions[:target_size - len(all_hint_versions)])
            else:
                # If still no hints, add empty ones
                all_hint_versions.extend([''] * (target_size - len(all_hint_versions)))
        
        # Truncate if too long
        all_hint_versions = all_hint_versions[:target_size]
        
        # Create expanded variants (NO original row, only expanded variants)
        expanded_variants = []
        current_index = start_index
        
        for i, hint in enumerate(all_hint_versions):
            variant = tsv_row.copy()
            variant['index'] = str(current_index)
            variant['hint'] = hint
            expanded_variants.append(variant)
            current_index += 1
        
        logger.debug(f"Created exactly {len(expanded_variants)} variants for index {record_index}")
        return expanded_variants, record_index

    def _process_tsv_records_parallel(self, all_records: List[Tuple], expanded_rows: List[Dict], 
                                    processed_indices: set, evolvement_types: List[str], 
                                    granularity: str, next_index: int) -> None:
        """Process TSV records in parallel using ThreadPoolExecutor."""
        import threading
        
        # Thread-safe index counter
        index_lock = threading.Lock()
        current_index = [next_index]  # Use list for mutable reference
        
        def get_next_index_batch(num_variants: int) -> int:
            with index_lock:
                start_idx = current_index[0]
                current_index[0] += num_variants
                return start_idx
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_record = {}
            
            for record_data in all_records:
                # Get a batch of indices for this record (estimate based on max possible variants)
                estimated_variants = 10 * len(evolvement_types)  # Reasonable estimate
                start_idx = get_next_index_batch(estimated_variants)
                
                future = executor.submit(self._process_single_tsv_record, record_data, 
                                       evolvement_types, granularity, start_idx)
                future_to_record[future] = record_data[2]  # record_index
            
            # Collect results with progress bar
            with tqdm(total=len(all_records), desc="Processing TSV records (parallel)") as pbar:
                for future in as_completed(future_to_record):
                    record_index = future_to_record[future]
                    try:
                        variants, _ = future.result()
                        
                        # Thread-safe updates
                        with index_lock:
                            expanded_rows.extend(variants)
                            processed_indices.add(record_index)
                        
                        logger.debug(f"Created {len(variants)} total variants for index {record_index}")
                        
                    except Exception as e:
                        logger.error(f"Error processing record {record_index}: {e}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'expanded_rows': len(expanded_rows)
                    })

    def _process_tsv_records_sequential(self, all_records: List[Tuple], expanded_rows: List[Dict], 
                                      processed_indices: set, evolvement_types: List[str], 
                                      granularity: str, next_index: int) -> None:
        """Process TSV records sequentially (original logic)."""
        current_index = next_index
        
        with tqdm(total=len(all_records), desc="Processing TSV records (sequential)") as pbar:
            for record_data in all_records:
                try:
                    variants, record_index = self._process_single_tsv_record(
                        record_data, evolvement_types, granularity, current_index)
                    
                    expanded_rows.extend(variants)
                    processed_indices.add(record_index)
                    current_index += len(variants)
                    
                    logger.debug(f"Created {len(variants)} total variants for index {record_index}")
            
                except Exception as e:
                    record_index = record_data[2]
                    logger.error(f"Error processing record {record_index}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'expanded_rows': len(expanded_rows)
                })

def main():
    """
    Main entry point for the CLI tool.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract layered academic hints from JSONL using an LLM")
    parser.add_argument("--in", dest="in_path", help="Path to input JSONL file (required unless using TSV expansion)")
    parser.add_argument("--out", dest="out_path", help="Path to output JSONL file (default: X.hints.jsonl)")
    parser.add_argument("--force-english", action="store_true", help="Translate hints to English")
    parser.add_argument("--api-key", help="LLM API key (fallback: MODEL_API_KEY env var)")
    parser.add_argument("--base-url", help="LLM API base URL (fallback: MODEL_API_BASE_URL env var)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0, help="LLM top-p (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--dry-run", type=int, help="Process first N lines to stdout")
    
    # New CLI flags for logic-tree guided hint generation
    parser.add_argument("--difficulty", choices=["minimal", "standard", "guided"], default="standard", 
                       help="Difficulty level for hints (default: standard)")
    parser.add_argument("--variants", type=int, default=1, 
                       help="Number of alternative hint sets per difficulty (default: 1)")
    parser.add_argument("--max-tiers", type=int, default=3, 
                       help="Maximum number of hint tiers (default: 3)")
    parser.add_argument("--style", choices=["socratic", "direct"], default="direct", 
                       help="Hint style: socratic (questions) or direct (statements) (default: direct)")
    parser.add_argument("--max_hint_chars", type=int, default=220, 
                       help="Maximum characters per hint (default: 220)")
    parser.add_argument("--num-workers", type=int, default=128, 
                       help="Number of worker threads for parallel LLM API calls (default: 128)")
    parser.add_argument("--hard_num", type=int, default=4, 
                       help="Number of misleading versions to generate for hard evolvement (default: 4)")
    parser.add_argument("--model", default="gpt-4", 
                       help="LLM model to use (default: gpt-4)")
    parser.add_argument("--timeout", type=float, default=60.0, 
                       help="Timeout for LLM calls in seconds (default: 60.0)")
    
    # TSV expansion flags
    parser.add_argument("--tsv-in", help="Path to input TSV file for expansion")
    parser.add_argument("--tsv-out", help="Path to output expanded TSV file")
    parser.add_argument("--evolvement", default="easy",
                       help="Evolvement type: 'easy' (helpful hints), 'hard' (misleading hints), or 'easy,hard' (both) (default: easy)")
    parser.add_argument("--expand-size", type=int, default=4,
                       help="Exact number of variants to generate per question (default: 4)")
    
    args = parser.parse_args()
    
    # Check for TSV expansion mode
    tsv_expansion_mode = bool(args.tsv_in and args.tsv_out)
    
    if tsv_expansion_mode:
        # TSV expansion mode: requires both tsv-in and tsv-out, and in_path for JSONL
        if not args.in_path:
            logger.error("TSV expansion mode requires --in (JSONL file) along with --tsv-in and --tsv-out")
            sys.exit(1)
        if not args.tsv_in:
            logger.error("TSV expansion mode requires --tsv-in")
            sys.exit(1)
        if not args.tsv_out:
            logger.error("TSV expansion mode requires --tsv-out")
            sys.exit(1)
    else:
        # Regular JSONL mode: requires in_path
        if not args.in_path:
            logger.error("Regular mode requires --in (JSONL file). For TSV expansion, provide --tsv-in and --tsv-out")
            sys.exit(1)
    
    # Get API credentials
    api_key = args.api_key or os.environ.get("MODEL_API_KEY")
    base_url = args.base_url or os.environ.get("MODEL_API_BASE_URL")
    
    if not api_key:
        logger.error("API key is required: provide --api-key or set MODEL_API_KEY env var")
        sys.exit(1)
        
    if not base_url:
        logger.error("Base URL is required: provide --base-url or set MODEL_API_BASE_URL env var")
        sys.exit(1)
    
    try:
        # Initialize LLM client
        client = init_client(api_key, base_url)
        
        # Initialize extractor with all new parameters
        extractor = GeoHintExtractor(
            client=client,
            max_hint_chars=args.max_hint_chars,
            force_english=args.force_english,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            model=args.model,
            difficulty=args.difficulty,
            variants=args.variants,
            max_tiers=args.max_tiers,
            style=args.style,
            timeout=args.timeout,
            evolvement=args.evolvement,
            num_workers=args.num_workers,
            hard_num=args.hard_num,
            expand_size=args.expand_size
        )
        
        if tsv_expansion_mode:
            # TSV expansion mode
            logger.info(f"Running TSV expansion mode")
            logger.info(f"JSONL input: {args.in_path}")
            logger.info(f"TSV input: {args.tsv_in}")
            logger.info(f"TSV output: {args.tsv_out}")
            logger.info(f"Granularity: hint (fixed)")
            if args.dry_run:
                logger.info(f"Dry run mode: processing first {args.dry_run} JSONL records")
            
            extractor.process_tsv_expansion(
                jsonl_path=args.in_path,
                tsv_in_path=args.tsv_in,
                tsv_out_path=args.tsv_out,
                dry_n=args.dry_run
            )
            
        else:
            # Regular JSONL processing mode
            in_path = args.in_path
            out_path = args.out_path
            
            if not out_path:
                # Derive output path if not specified
                in_path_obj = Path(in_path)
                if in_path_obj.suffix.lower() == '.jsonl':
                    out_path = str(in_path_obj.with_name(in_path_obj.stem + '_hints.jsonl'))
                else:
                    out_path = f"{in_path}_hints.json"
            
            # Process the input file
            logger.info(f"Processing {in_path}")
            if args.dry_run:
                logger.info(f"Dry run mode: processing first {args.dry_run} records")
            else:
                logger.info(f"Output will be written to {out_path}")
                
            if args.num_workers > 1:
                logger.info(f"Using {args.num_workers} worker threads")
                extractor.process_stream_concurrent(in_path, out_path, args.dry_run)
            else:
                extractor.process_stream(in_path, out_path, args.dry_run)
        
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()