#!/usr/bin/env python3.10
"""Collect model features from name parsing, LMArena, HuggingFace, and OpenRouter."""

import re
import time
import math
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = "/Users/antoinemaechler/Documents/dr_benchmark_biased/cs321m_project"

# ── Provider detection ──────────────────────────────────────────────────────

PROVIDER_PATTERNS = [
    # Check specific patterns before generic keyword matching
    ("openai",    [r"(?<![a-zA-Z])gpt[-_]", r"\bo[134][-_ ]", r"\bGPT[- ]?[45]", r"GPT OSS", r"(?i)chatgpt", r"openai/", r"(?<![a-zA-Z])gpt-?4o"]),
    ("anthropic", [r"claude", r"Claude", r"opus", r"sonnet", r"haiku"]),
    ("google",    [r"gemini", r"Gemini", r"gemma", r"Gemma", r"palm", r"bard", r"PaLM", r"paligemma",
                   r"medlm", r"medpalm", r"google/"]),
    ("meta",      [r"(?i)llama", r"Meta-", r"meta-llama", r"meta[-_]llama"]),
    ("mistral",   [r"[Mm]istral", r"[Mm]ixtral", r"[Mm]athstral", r"[Cc]odestral", r"[Mm]inistral",
                   r"mistralai/", r"Magistral"]),
    ("deepseek",  [r"[Dd]eep[Ss]eek", r"[Dd]eep[Cc]oder", r"DSCoder"]),
    ("qwen",      [r"[Qq]wen", r"QwQ", r"qwq", r"Qwen"]),
    ("microsoft", [r"[Pp]hi[-_]", r"Phi-"]),
    ("cohere",    [r"command-r", r"c4ai-command", r"Cohere"]),
    ("xai",       [r"[Gg]rok"]),
    ("zhipu",     [r"glm[-_]", r"GLM", r"chatglm", r"zhipu"]),
    ("minimax",   [r"abab", r"[Mm]ini[Mm]ax", r"MiniMax"]),
    ("baidu",     [r"baidu", r"ernie"]),
    ("alibaba",   [r"alibaba"]),
    ("nvidia",    [r"[Nn]emotron", r"NVLM", r"nvidia"]),
]

# Models that are closed-source (not just by provider)
CLOSED_PROVIDERS = {"openai", "anthropic", "google", "xai", "minimax", "baidu",
                    "zhipu", "cohere", "other_closed"}
CLOSED_PATTERNS = [
    r"(?<![a-zA-Z])gpt[-_]", r"\bo[134][-_ ]", r"(?<![a-zA-Z])GPT[- ]?[45]", r"GPT OSS", r"(?<![a-zA-Z])gpt-?4o",
    r"claude", r"Claude", r"opus", r"sonnet", r"haiku",
    r"gemini", r"Gemini", r"palm", r"bard",
    r"[Gg]rok",
    r"abab", r"MiniMax", r"MiniMax",
    r"glm[-_]4", r"GLM [45]",
    r"medlm", r"medpalm",
    r"command-r",
    r"moonshot",
    r"[Rr]eka",
    r"[Ss]tep1",
    r"Kimi",
]

# Architecture family
ARCH_PATTERNS = [
    ("gpt",       [r"(?<![a-zA-Z])gpt[-_]", r"\bo[134][-_ ]", r"(?<![a-zA-Z])GPT[- ]?[45]", r"GPT OSS", r"(?i)chatgpt", r"(?<![a-zA-Z])gpt-?4o"]),
    ("claude",    [r"claude", r"Claude", r"opus", r"sonnet", r"haiku"]),
    ("gemini",    [r"gemini", r"Gemini"]),
    ("llama",     [r"(?i)llama", r"vicuna", r"alpaca", r"guanaco",
                   r"[Bb]aize", r"koala", r"tulu"]),
    ("mistral",   [r"[Mm]istral", r"[Mm]ixtral", r"[Mm]athstral", r"[Cc]odestral",
                   r"[Mm]inistral", r"zephyr"]),
    ("qwen",      [r"[Qq]wen", r"QwQ", r"qwq"]),
    ("deepseek",  [r"[Dd]eep[Ss]eek", r"[Dd]eep[Cc]oder", r"DSCoder"]),
    ("phi",       [r"[Pp]hi[-_]", r"[Pp]hi\d"]),
    ("gemma",     [r"[Gg]emma"]),
    ("yi",        [r"\bYi[-_]", r"\byi[-_]"]),
    ("falcon",    [r"[Ff]alcon"]),
    ("internvl",  [r"[Ii]ntern[Vv][Ll]", r"[Ii]ntern[Ll][Mm]"]),
    ("llava",     [r"[Ll]la[Vv][Aa]"]),
]

# Multimodal patterns
MULTIMODAL_PATTERNS = [
    r"\bVL\b", r"[-_]VL[-_]", r"VL[-_]", r"[-_]VL\b",
    r"[Vv]ision", r"\bVLM\b",
    r"[Ll]la[Vv][Aa]", r"[Ii]ntern[Vv][Ll]",
    r"[Cc]og[Vv][Ll][Mm]", r"[Ii]defics", r"[Pp]ali[Gg]emma",
    r"[Ff]lamingo", r"[Mm]olmo", r"[Cc]ambrian",
    r"[Ee]agle", r"[Bb]unny", r"[Mm]antis",
    r"[Xx][Cc]omposer", r"[Mm]oondream",
    r"[Mm]ini[Gg][Pp][Tt]", r"[Ii]nstruct[Bb]lip", r"m[Pp][Ll][Uu][Gg]",
    r"[Ss]mol[Vv][Ll][Mm]", r"\b[Qq][Vv][Qq]\b",
    r"[Oo]vis", r"\bmm\b", r"[Oo]mni[Ll][Mm][Mm]",
    r"GPT4[Vv]", r"GPT4o",  # Vision versions of GPT
    r"GeminiPro[Vv]ision",
    r"Claude3[-_]?[0-9]*V", r"[Bb]ailing[Mm][Mm]",
    r"[Dd]oubao[Vv][Ll]", r"[Ss]ense[Cc]hat",
    r"[Mm]ini[Cc][Pp][Mm].*V", r"[Mm]ini[Cc][Pp][Mm]-[Vv]",
    r"MiniCPM-o", r"[Pp]ixtral",
    r"[Kk]osmos", r"[Ss]hare[Gg][Pp][Tt]4[Vv]",
    r"[Ss]hare[Cc]aptioner",
    r"[Hh]un[Yy]uan.*[Vv]ision", r"[Hh]un[Yy]uan.*Standard.*Vision",
    r"emu[23]", r"chameleon", r"monkey",
    r"[Vv]intera?n", r"VILA", r"vita",
    r"xgen-mm", r"RBDash", r"Taichu",
    r"[Tt]ele[Mm][Mm]", r"[Ss]lime", r"[Aa]qua",
    r"[Vv]arco.*vision", r"BlueLM_V",
    r"Falcon2-VLM", r"GLM4V", r"GLM 4\.5V",
    r"360VL", r"POINTS", r"POINTSV",
    r"SAIL-VL", r"WeMM", r"CloudWalk",
    r"ross-qwen", r"VXVERSE", r"qwen_base", r"qwen_chat",
    r"Qwen.*VL",
    r"Aria", r"[Oo]la\b", r"OmChat",
    r"JTVL", r"[Jj]anus", r"Ristretto",
    r"MMAlaya", r"MUG-U", r"URSA",
    r"Parrot\b", r"TransCore",
    r"[Vv]alley_eagle", r"XinYuan-VL",
    r"Taiyi\b", r"h2ovl",
    r"Yi[_-]Vision", r"Yi_VL",
    r"LLaVA-CoT", r"VLM-R1", r"VLAA",
    r"Qwen.*VL.*Thinking",
]


def parse_param_count(name: str) -> float:
    """Extract parameter count in billions from model name."""
    # Handle MoE patterns like 8x7B, 8x22B
    moe = re.search(r'(\d+)x(\d+)[Bb]', name)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))

    # Handle patterns like 235B-A22B → 235
    total_with_active = re.search(r'(\d+)[Bb][-_]A\d+[Bb]', name)
    if total_with_active:
        return float(total_with_active.group(1))

    # Handle pythia models: pythia-12b, pythia12-0b → 12 (not 0)
    pythia = re.search(r'pythia[-_]?(\d+)(?:[-._]\d+)?[Bb]', name, re.IGNORECASE)
    if pythia:
        return float(pythia.group(1))

    # Find ALL occurrences of number+B pattern and pick the right one
    # Use word-boundary-aware pattern to avoid matching version numbers
    # Pattern: a number (possibly with . or _) followed by B, where B is at a word boundary
    candidates = []
    for m in re.finditer(r'(?<![.\d])(\d+(?:[._]\d+)?)\s*[Bb]\b', name):
        val_str = m.group(1).replace('_', '.').replace('-', '.')
        val = float(val_str)
        candidates.append((val, m.start()))

    if candidates:
        # Prefer the largest value that looks like a param count (not a version)
        # Check context: skip if preceded by "v" or "V" (version) or if it's a date
        best = None
        for val, pos in candidates:
            # Check if this is preceded by "v" or "V" (version number)
            if pos > 0 and name[pos-1] in ('v', 'V'):
                continue
            if best is None or val > best:
                best = val
        if best is not None and best > 0:
            return best

    return np.nan


def detect_provider(name: str) -> str:
    """Detect provider family from model name."""
    low = name.lower()

    # SWE-bench agents: check for embedded provider hints
    if re.match(r'^\d{8}_', name):
        if 'claude' in low or 'sonnet' in low or 'opus' in low or 'haiku' in low:
            return 'anthropic'
        if 'gpt' in low or re.search(r'o[134][-_ ]', low) or 'o4-mini' in low:
            return 'openai'
        if 'gemini' in low:
            return 'google'
        if 'llama' in low:
            return 'meta'
        if 'qwen' in low or 'lingma' in low:
            return 'qwen'
        if 'deepseek' in low:
            return 'deepseek'
        if 'mistral' in low or 'devstral' in low:
            return 'mistral'
        if 'kimi' in low:
            return 'other_closed'
        if 'glm' in low:
            return 'zhipu'
        if 'gpt5' in low or 'gpt-5' in low:
            return 'openai'
        if 'nova' in low:
            return 'other_closed'
        return 'other_open'

    # Org prefix detection
    org = name.split('/')[0].lower() if '/' in name else ''
    if org in ('openai',):
        return 'openai'
    if org in ('meta-llama',):
        return 'meta'
    if org in ('google',):
        return 'google'
    if org in ('mistralai',):
        return 'mistral'
    if org in ('deepseek-ai',):
        return 'deepseek'
    if org in ('qwen',):
        return 'qwen'

    for provider, patterns in PROVIDER_PATTERNS:
        for pat in patterns:
            if re.search(pat, name):
                return provider

    # Some specific models
    if 'reka' in low:
        return 'other_closed'
    if 'moonshot' in low:
        return 'other_closed'
    if 'step1' in low or 'Step ' in name:
        return 'other_closed'
    if 'kimi' in low or 'Kimi' in name:
        return 'other_closed'
    if 'jamba' in low:
        return 'other_open'
    if 'rwkv' in low or 'mpt' in low or 'falcon' in low:
        return 'other_open'
    if 'stablelm' in low or 'starchat' in low or 'stable-code' in low or 'stabilityai/' in name:
        return 'other_open'
    if 'nous' in low or 'hermes' in low:
        return 'other_open'
    if 'solar' in low or 'upstage/' in name:
        return 'other_open'
    if 'olmo' in low or 'allenai/' in name:
        return 'other_open'
    if 'yi' in low and ('01.ai' in low or 'Yi-' in name or 'Yi_' in name):
        return 'other_open'
    if 'pythia' in low or 'dolly' in low:
        return 'other_open'
    if 'internlm' in low or 'internvl' in low:
        return 'other_open'
    if 'skywork' in low:
        return 'other_open'
    if 'baichuan' in low:
        return 'other_open'
    if re.search(r'[Dd]oubao', name):
        return 'other_closed'
    if 'iask' in low:
        return 'other_closed'
    if 'AzeroGPT' in name:
        return 'other_closed'
    if 'EXAONE' in name:
        return 'other_open'
    if 'QED' in name:
        return 'other_open'
    if 'LIMO' in name or 's1.1' in name:
        return 'other_open'

    # Check if it has org/ prefix → likely open source
    if '/' in name:
        return 'other_open'

    return 'other_open'


def detect_arch(name: str) -> str:
    """Detect architecture family from model name."""
    # SWE-bench agents: check embedded model names
    low = name.lower()
    if re.match(r'^\d{8}_', name):
        if 'claude' in low or 'sonnet' in low or 'opus' in low or 'haiku' in low:
            return 'claude'
        if 'gpt' in low:
            return 'gpt'
        if 'gemini' in low:
            return 'gemini'
        if 'llama' in low:
            return 'llama'
        if 'qwen' in low:
            return 'qwen'
        if 'deepseek' in low:
            return 'deepseek'
        if 'mistral' in low or 'devstral' in low:
            return 'mistral'
        return 'other'

    for arch, patterns in ARCH_PATTERNS:
        for pat in patterns:
            if re.search(pat, name):
                return arch
    # Some special cases
    if re.search(r'\bo[134]-', name) or re.search(r'GPT[- ]?[45]', name) or 'GPT OSS' in name:
        return 'gpt'
    if 'rwkv' in low:
        return 'other'
    if 'mpt' in low:
        return 'other'
    if 'pythia' in low:
        return 'other'
    if 'stablelm' in low:
        return 'other'
    if 'falcon' in low:
        return 'falcon'
    return 'other'


def parse_name_features(display_name: str) -> dict:
    """Extract all name-based features from a model display name."""
    name = display_name
    low = name.lower()

    # Parameter count
    param_count = parse_param_count(name)
    log_param = np.log10(param_count) if not np.isnan(param_count) else np.nan

    # Provider and architecture
    provider = detect_provider(name)
    arch = detect_arch(name)

    # Is instruct-tuned
    is_instruct = bool(re.search(
        r'[Ii]nstruct|[Cc]hat|[Dd][Pp][Oo]|[Rr][Ll][Hh][Ff]|[-_][Ii][Tt]\b|[-_]Ins\b|[Ss][Ff][Tt]',
        name
    ))

    # Is multimodal/vision
    is_multimodal = False
    for pat in MULTIMODAL_PATTERNS:
        if re.search(pat, name):
            is_multimodal = True
            break

    # Is closed-source
    is_closed = provider in CLOSED_PROVIDERS
    if not is_closed:
        for pat in CLOSED_PATTERNS:
            if re.search(pat, name):
                is_closed = True
                break

    # Is SWE-bench agent
    is_swebench = bool(re.match(r'^\d{8}_', name))

    # Is reward model
    is_reward = bool(re.search(
        r'[Rr]eward|RM[-_]|[-_][Rr][Mm][-_]|[-_][Rr][Mm]$|[Cc]ost|[Cc]ritic',
        name
    ))

    # Is FC variant
    is_fc = name.endswith('-FC')

    # Is reasoning
    is_reasoning = bool(re.search(
        r'[Tt]hink|[Oo]1[-_ ]|[Oo]3[-_ ]|[Oo]4[-_ ]|\bo1\b|\bo3\b|\bo4\b|'
        r'QwQ|qwq|[Rr]1\b|[Rr]easoning|DeepThink|'
        r'o1-pro|O1-|O3[-_ ]|O4[-_ ]',
        name
    ))

    return {
        'param_count_b': param_count,
        'log_param_count': log_param,
        'provider': provider,
        'arch_family': arch,
        'is_instruct': int(is_instruct),
        'is_multimodal': int(is_multimodal),
        'is_closed': int(is_closed),
        'is_swebench_agent': int(is_swebench),
        'is_reward_model': int(is_reward),
        'is_fc_variant': int(is_fc),
        'is_reasoning': int(is_reasoning),
    }


# ── External data sources ──────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Normalize a model name for fuzzy matching."""
    name = name.lower().strip()
    # Strip org prefixes
    if '/' in name:
        name = name.split('/')[-1]
    # Remove date suffixes like -20240620, -2024-04-09
    name = re.sub(r'[-_]\d{4}[-_]?\d{2}[-_]?\d{2}$', '', name)
    # Remove version suffixes like -v0.1, -v2
    name = re.sub(r'[-_]v\d+(\.\d+)*$', '', name)
    # Normalize separators
    name = re.sub(r'[-_. ]+', '-', name)
    return name


def fetch_arena_scores() -> pd.DataFrame:
    """Fetch LMArena leaderboard and return DataFrame with elo, rank, votes."""
    print("  Fetching LMArena scores...")
    try:
        from datasets import load_dataset
        ds = load_dataset("lmarena-ai/leaderboard-dataset", "text", split="latest")
        arena = ds.to_pandas()
        arena_overall = arena[arena['category'] == 'overall'][['model_name', 'rating', 'vote_count', 'rank']].copy()
        arena_overall.columns = ['arena_name', 'arena_elo', 'arena_votes', 'arena_rank']
        print(f"  Got {len(arena_overall)} arena models")
        return arena_overall
    except Exception as e:
        print(f"  datasets failed ({e}), trying REST API...")
        try:
            import requests
            resp = requests.get("https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name=text", timeout=30)
            data = resp.json()
            rows = []
            for item in data:
                rows.append({
                    'arena_name': item.get('model_name', ''),
                    'arena_elo': item.get('rating', np.nan),
                    'arena_votes': item.get('vote_count', np.nan),
                    'arena_rank': item.get('rank', np.nan),
                })
            df = pd.DataFrame(rows)
            print(f"  Got {len(df)} arena models from REST API")
            return df
        except Exception as e2:
            print(f"  REST API also failed: {e2}")
            return pd.DataFrame(columns=['arena_name', 'arena_elo', 'arena_votes', 'arena_rank'])


def match_arena_to_models(models_df: pd.DataFrame, arena_df: pd.DataFrame) -> pd.DataFrame:
    """Match arena model names to our model display names."""
    if arena_df.empty:
        return pd.DataFrame(index=models_df.index, columns=['arena_elo', 'arena_rank', 'arena_votes'])

    # Build normalized lookup from arena
    arena_lookup = {}
    for _, row in arena_df.iterrows():
        key = normalize_name(row['arena_name'])
        arena_lookup[key] = row

    results = []
    for _, mrow in models_df.iterrows():
        name = mrow['display_name']
        norm = normalize_name(name)

        match = None
        # Exact normalized match
        if norm in arena_lookup:
            match = arena_lookup[norm]
        else:
            # Try stripping -fc suffix
            stripped = re.sub(r'-fc$', '', norm)
            if stripped in arena_lookup:
                match = arena_lookup[stripped]
            else:
                # Try with common variations
                for arena_key, arena_row in arena_lookup.items():
                    if norm == arena_key or stripped == arena_key:
                        match = arena_row
                        break
                    # Check if one is substring of the other (for short names)
                    if len(norm) > 5 and len(arena_key) > 5:
                        if norm in arena_key or arena_key in norm:
                            match = arena_row
                            break

        if match is not None:
            results.append({
                'arena_elo': match['arena_elo'],
                'arena_rank': match['arena_rank'],
                'arena_votes': match['arena_votes'],
            })
        else:
            results.append({
                'arena_elo': np.nan,
                'arena_rank': np.nan,
                'arena_votes': np.nan,
            })

    return pd.DataFrame(results, index=models_df.index)


def fetch_hf_metadata(display_names: list) -> pd.DataFrame:
    """Fetch HuggingFace metadata for matchable models."""
    print("  Fetching HuggingFace metadata...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except Exception as e:
        print(f"  huggingface_hub not available: {e}")
        return pd.DataFrame(columns=['display_name', 'hf_downloads', 'hf_likes', 'hf_created_at'])

    results = []
    # Models with org/ prefix → direct lookup
    direct_names = [n for n in display_names if '/' in n and not n.endswith('/')]
    other_open = [n for n in display_names if '/' not in n]

    count = 0
    for name in direct_names:
        try:
            info = api.model_info(name)
            results.append({
                'display_name': name,
                'hf_downloads': getattr(info, 'downloads', np.nan),
                'hf_likes': getattr(info, 'likes', np.nan),
                'hf_created_at': str(getattr(info, 'created_at', ''))[:10] if getattr(info, 'created_at', None) else np.nan,
            })
            count += 1
        except Exception:
            pass
        time.sleep(0.5)

    # For non-org models, try search with well-known prefixes
    KNOWN_HF_PREFIXES = {
        'Meta-Llama': 'meta-llama',
        'Llama-2': 'meta-llama',
        'Llama-3': 'meta-llama',
        'Mixtral': 'mistralai',
        'Mistral': 'mistralai',
        'Phi-': 'microsoft',
        'Qwen': 'Qwen',
        'Yi-': '01-ai',
        'gemma': 'google',
        'Gemma': 'google',
        'Falcon': 'tiiuae',
        'BioMistral': 'BioMistral',
        'Meditron': 'epfl-llm',
    }

    for name in other_open:
        if re.match(r'^\d{8}_', name):  # skip SWE-bench agents
            continue
        # Try known prefix mapping
        repo = None
        for prefix, org in KNOWN_HF_PREFIXES.items():
            if name.startswith(prefix):
                repo = f"{org}/{name}"
                break

        if repo:
            try:
                info = api.model_info(repo)
                results.append({
                    'display_name': name,
                    'hf_downloads': getattr(info, 'downloads', np.nan),
                    'hf_likes': getattr(info, 'likes', np.nan),
                    'hf_created_at': str(getattr(info, 'created_at', ''))[:10] if getattr(info, 'created_at', None) else np.nan,
                })
                count += 1
            except Exception:
                # Try search as fallback
                try:
                    hits = list(api.list_models(search=name, limit=1))
                    if hits and name.lower() in hits[0].id.lower():
                        info = hits[0]
                        results.append({
                            'display_name': name,
                            'hf_downloads': getattr(info, 'downloads', np.nan),
                            'hf_likes': getattr(info, 'likes', np.nan),
                            'hf_created_at': str(getattr(info, 'created_at', ''))[:10] if getattr(info, 'created_at', None) else np.nan,
                        })
                        count += 1
                except Exception:
                    pass
            time.sleep(0.5)

    print(f"  Got HF metadata for {count} models")
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['display_name', 'hf_downloads', 'hf_likes', 'hf_created_at'])


def fetch_openrouter_data() -> pd.DataFrame:
    """Fetch OpenRouter model catalog."""
    print("  Fetching OpenRouter data...")
    try:
        import requests
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        models = resp.json().get("data", [])
        rows = []
        for m in models:
            model_id = m.get("id", "")
            # Normalize: strip org prefix
            short_name = model_id.split("/")[-1] if "/" in model_id else model_id
            ctx = m.get("context_length", np.nan)
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0)) if pricing.get("prompt") else np.nan
            rows.append({
                'or_name': short_name,
                'or_full_id': model_id,
                'openrouter_context_length': ctx,
                'openrouter_prompt_price': prompt_price,
            })
        df = pd.DataFrame(rows)
        print(f"  Got {len(df)} OpenRouter models")
        return df
    except Exception as e:
        print(f"  OpenRouter fetch failed: {e}")
        return pd.DataFrame(columns=['or_name', 'or_full_id', 'openrouter_context_length', 'openrouter_prompt_price'])


def match_openrouter_to_models(models_df: pd.DataFrame, or_df: pd.DataFrame) -> pd.DataFrame:
    """Match OpenRouter models to our models."""
    if or_df.empty:
        return pd.DataFrame(index=models_df.index, columns=['openrouter_context_length', 'openrouter_prompt_price'])

    # Build lookup
    or_lookup = {}
    for _, row in or_df.iterrows():
        key = normalize_name(row['or_name'])
        or_lookup[key] = row
        # Also index by full ID normalized
        key2 = normalize_name(row['or_full_id'])
        or_lookup[key2] = row

    results = []
    for _, mrow in models_df.iterrows():
        name = mrow['display_name']
        norm = normalize_name(name)

        match = None
        if norm in or_lookup:
            match = or_lookup[norm]
        else:
            stripped = re.sub(r'-fc$', '', norm)
            if stripped in or_lookup:
                match = or_lookup[stripped]
            else:
                for or_key, or_row in or_lookup.items():
                    if len(norm) > 5 and len(or_key) > 5:
                        if norm in or_key or or_key in norm:
                            match = or_row
                            break

        if match is not None:
            results.append({
                'openrouter_context_length': match['openrouter_context_length'],
                'openrouter_prompt_price': match['openrouter_prompt_price'],
            })
        else:
            results.append({
                'openrouter_context_length': np.nan,
                'openrouter_prompt_price': np.nan,
            })

    return pd.DataFrame(results, index=models_df.index)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Collecting model features")
    print("=" * 60)

    # 1. Load subjects
    print("\n1. Loading subjects.parquet...")
    subjects = pd.read_parquet(f"{SCRIPT_DIR}/data/subjects.parquet")
    print(f"   {len(subjects)} models loaded")

    # 2. Parse name features
    print("\n2. Parsing name features...")
    features = subjects['display_name'].apply(parse_name_features).apply(pd.Series)
    df = pd.concat([subjects[['subject_id', 'display_name']], features], axis=1)
    print(f"   Parsed features for {len(df)} models")

    # 3. Fetch arena scores
    print("\n3. Fetching external data...")
    arena_df = fetch_arena_scores()
    arena_matched = match_arena_to_models(df, arena_df)
    df = pd.concat([df, arena_matched], axis=1)
    print(f"   Arena matches: {arena_matched['arena_elo'].notna().sum()}")

    # 4. Fetch HF metadata
    hf_df = fetch_hf_metadata(subjects['display_name'].tolist())
    if not hf_df.empty:
        hf_df = hf_df.drop_duplicates(subset='display_name', keep='first')
        df = df.merge(hf_df, on='display_name', how='left')
    else:
        df['hf_downloads'] = np.nan
        df['hf_likes'] = np.nan
        df['hf_created_at'] = np.nan
    print(f"   HF matches: {df['hf_downloads'].notna().sum()}")

    # 5. Fetch OpenRouter data
    or_df = fetch_openrouter_data()
    or_matched = match_openrouter_to_models(df, or_df)
    # Reset index alignment
    or_matched.index = df.index
    df['openrouter_context_length'] = or_matched['openrouter_context_length'].values
    df['openrouter_prompt_price'] = or_matched['openrouter_prompt_price'].values
    print(f"   OpenRouter matches: {df['openrouter_context_length'].notna().sum()}")

    # 6. Ensure column order
    cols = [
        'subject_id', 'display_name', 'param_count_b', 'log_param_count',
        'provider', 'arch_family', 'is_instruct', 'is_multimodal', 'is_closed',
        'is_swebench_agent', 'is_reward_model', 'is_fc_variant', 'is_reasoning',
        'arena_elo', 'arena_rank', 'arena_votes',
        'hf_downloads', 'hf_likes', 'hf_created_at',
        'openrouter_context_length', 'openrouter_prompt_price',
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    # 7. Save
    out_path = f"{SCRIPT_DIR}/model_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n   Saved to {out_path}")

    # 8. Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(f"\nTotal models: {len(df)}")

    print("\nFeature coverage (non-null counts):")
    for col in cols[2:]:
        n = df[col].notna().sum()
        pct = n / len(df) * 100
        print(f"  {col:30s}: {n:4d} / {len(df)} ({pct:.1f}%)")

    print("\nProvider distribution:")
    print(df['provider'].value_counts().to_string())

    print("\nArchitecture family distribution:")
    print(df['arch_family'].value_counts().to_string())

    print("\nSample rows for well-known models:")
    sample_names = [
        'gpt-4o', 'claude-3-5-sonnet-20241022', 'Meta-Llama-3-70B-Instruct',
        'Mixtral-8x7B-Instruct-v0.1', 'deepseek-r1', 'qwq-32b',
        'gemini-2.0-flash', 'Phi-3-mini-128k-instruct',
        'gpt-4o-mini-2024-07-18-FC', '20240620_sweagent_claude3.5sonnet',
    ]
    for sn in sample_names:
        row = df[df['display_name'] == sn]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  {sn}:")
            print(f"    provider={r['provider']}, arch={r['arch_family']}, "
                  f"params={r['param_count_b']}, instruct={r['is_instruct']}, "
                  f"closed={r['is_closed']}, multimodal={r['is_multimodal']}, "
                  f"swebench={r['is_swebench_agent']}, fc={r['is_fc_variant']}, "
                  f"reasoning={r['is_reasoning']}")

    # Boolean summaries
    print("\nBoolean feature sums:")
    for col in ['is_instruct', 'is_multimodal', 'is_closed', 'is_swebench_agent',
                'is_reward_model', 'is_fc_variant', 'is_reasoning']:
        print(f"  {col}: {int(df[col].sum())}")


if __name__ == "__main__":
    main()
