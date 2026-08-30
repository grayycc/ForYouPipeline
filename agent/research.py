"""Literature search for the planner, backed by OpenAlex.

OpenAlex rather than arXiv or Semantic Scholar: both of those return HTTP 429 from this
network. The planner finds its own papers instead of being handed a curated library, since
curating one would mean we chose the solution space.

Nothing here raises -- a search outage costs citation quality, never an iteration.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

from . import config

_ENDPOINT = 'https://api.openalex.org/works'
_FIELDS = 'id,doi,display_name,publication_year,authorships,abstract_inverted_index,cited_by_count'

TOOL_SCHEMA = {
    'name': 'search_papers',
    'description': (
        'Search the published literature for methods relevant to this task. Returns titles, '
        'authors, year, citation count, abstract, and a stable identifier for each paper. '
        'You may only cite identifiers returned by this tool. Search when you want to ground '
        'or check a mechanism; it is not required every iteration.'),
    'input_schema': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Search terms, e.g. "listwise ranking loss recommendation".',
            },
        },
        'required': ['query'],
    },
}


def _cache_path(query: str) -> str:
    """Cache file for a query, keyed on its normalised text so spacing and case do not miss."""
    key = hashlib.sha1(' '.join(query.lower().split()).encode()).hexdigest()
    return os.path.join(config.RESEARCH_CACHE_DIR, f'{key}.json')


def _rebuild_abstract(inverted) -> str:
    """OpenAlex stores abstracts as {word: [positions]}, not text."""
    if not isinstance(inverted, dict):
        return ''
    positions = [(i, word) for word, idxs in inverted.items() for i in idxs]
    return ' '.join(word for _, word in sorted(positions))


def _shape(work: dict) -> dict:
    """One OpenAlex record trimmed to the fields the planner reads, abstract truncated."""
    abstract = _rebuild_abstract(work.get('abstract_inverted_index'))
    if len(abstract) > config.ABSTRACT_CHARS:
        abstract = abstract[:config.ABSTRACT_CHARS].rsplit(' ', 1)[0] + '...'
    return {
        'id': (work.get('id') or '').rsplit('/', 1)[-1],
        'doi': (work.get('doi') or '').replace('https://doi.org/', ''),
        'title': work.get('display_name') or '',
        'authors': [a['author']['display_name']
                    for a in (work.get('authorships') or [])[:3]
                    if a.get('author', {}).get('display_name')],
        'year': work.get('publication_year'),
        'cited_by_count': work.get('cited_by_count'),
        'abstract': abstract,
    }


def search_papers(query: str, max_results: int = None) -> Dict:
    """Search OpenAlex. Returns {'query', 'results', 'cached'} or {'error'}. Never raises."""
    max_results = max_results or config.PAPERS_PER_SEARCH
    if not config.RESEARCH_ENABLED:
        return {'query': query, 'error': 'literature search is disabled for this run',
                'results': []}
    if not query or not query.strip():
        return {'query': query, 'error': 'empty query', 'results': []}

    path = _cache_path(query)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return {'query': query, 'cached': True, 'results': json.load(fh)[:max_results]}
        except Exception:
            pass  # unreadable cache entry: fall through and refetch

    url = _ENDPOINT + '?' + urllib.parse.urlencode({
        'search': query, 'per-page': max_results,
        'mailto': config.OPENALEX_MAILTO, 'select': _FIELDS,
    })
    req = urllib.request.Request(url, headers={
        'User-Agent': f'KuaiRandResearchAgent/1.0 (mailto:{config.OPENALEX_MAILTO})'})

    try:
        raw = urllib.request.urlopen(req, timeout=config.RESEARCH_TIMEOUT_S).read()
        results = [_shape(w) for w in json.loads(raw).get('results', [])]
    except urllib.error.HTTPError as e:
        return {'query': query, 'error': f'search unavailable (HTTP {e.code})', 'results': []}
    except Exception as e:
        return {'query': query, 'error': f'search unavailable ({type(e).__name__})',
                'results': []}

    try:
        os.makedirs(config.RESEARCH_CACHE_DIR, exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(results, fh)
    except Exception:
        pass  # caching is an optimisation, not a requirement

    return {'query': query, 'cached': False, 'results': results}


def format_results(payload: Dict, seen: Dict[str, str] = None) -> str:
    """Render results for the model. `seen` maps id -> what happened when it was used before,
    so a paper already acted on carries its outcome rather than a bare 'seen' flag: one whose
    idea worked is a reason to go deeper, one that landed in the noise floor is a reason to
    move on."""
    if payload.get('error'):
        return (f"search for {payload['query']!r} failed: {payload['error']}. "
                f"Proceed without literature for this experiment.")
    if not payload.get('results'):
        return f"no results for {payload['query']!r}. Try different terms or proceed without."

    seen = seen or {}
    lines = [f"results for {payload['query']!r}"
             + (' (cached)' if payload.get('cached') else '')]
    for r in payload['results']:
        authors = ', '.join(r['authors']) + (' et al.' if len(r['authors']) == 3 else '')
        lines.append(
            f"\n[{r['id']}] {r['title']}\n"
            f"  {authors or 'unknown'} ({r['year']}), cited {r['cited_by_count']} times"
            + (f"\n  doi: {r['doi']}" if r['doi'] else '')
            + (f"\n  ALREADY USED: {seen[r['id']]}" if r['id'] in seen else '')
            + (f"\n  {r['abstract']}" if r['abstract'] else ''))
    lines.append('\nCite papers by the bracketed identifier. Citing anything not returned '
                 'by a search this run will be rejected.')
    return '\n'.join(lines)
