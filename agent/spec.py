"""The ExperimentSpec: the contract between the planner and the coder.

It separates what is being tested from how it gets written. A spec must state a mechanism and
a falsification condition, so a vague suggestion cannot become an experiment.

Parsing never raises -- a malformed spec is replanned, not crashed on.
"""
import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_TEXT_FIELDS = ('hypothesis', 'mechanism', 'proposed_change',
                        'expected_result', 'falsification_condition', 'target_metric')


@dataclasses.dataclass
class ExperimentSpec:
    hypothesis: str = ''
    mechanism: str = ''
    evidence: Dict[str, str] = dataclasses.field(default_factory=dict)
    target_metric: str = ''
    proposed_change: str = ''
    expected_result: str = ''
    falsification_condition: str = ''
    risks: Dict[str, str] = dataclasses.field(default_factory=dict)
    implementation_scope: List[str] = dataclasses.field(default_factory=list)
    tags: List[str] = dataclasses.field(default_factory=list)
    candidates_considered: List[str] = dataclasses.field(default_factory=list)
    reflection: str = ''
    # Which of 'mechanism' / 'implementation' / 'noise' / 'bug' the previous failure disproved.
    # Recorded because retiring a mechanism on one bad implementation is unrecoverable: the
    # rest of the run never revisits it, and the log should show that call being made.
    prior_failure_attribution: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict for the node record and log.jsonl."""
        return dataclasses.asdict(self)

    @property
    def cited_ids(self) -> List[str]:
        """Identifiers the Planner claims to be citing, from evidence.literature."""
        lit = self.evidence.get('literature', '')
        if isinstance(lit, list):
            lit = ' '.join(str(x) for x in lit)
        return _extract_ids(str(lit))

    def summary(self, width: int = 150) -> str:
        """One truncated line naming the change, for the console."""
        return (self.proposed_change or self.hypothesis)[:width].replace('\n', ' ')


def _extract_ids(text: str) -> List[str]:
    """Pull OpenAlex work ids (W123...) and DOIs out of free text."""
    import re
    ids = re.findall(r'\bW\d{6,}\b', text)
    ids += [d.rstrip('.,);') for d in re.findall(r'10\.\d{4,9}/\S+', text)]
    seen, out = set(), []
    for i in ids:
        k = i.lower()
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


def _largest_json_object(text: str) -> Optional[str]:
    """Return the longest balanced {...} span. Regex can't match nested braces reliably."""
    best, stack = None, []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            if not stack:
                cand = text[start:i + 1]
                if best is None or len(cand) > len(best):
                    best = cand
    return best


def parse_spec(text: str) -> Tuple[Optional[ExperimentSpec], str]:
    """Parse a Planner response into a spec. Returns (spec, error); never raises."""
    if not text or not text.strip():
        return None, 'empty planner response'

    from .executor import strip_fences
    blob = _largest_json_object(strip_fences(text)) or _largest_json_object(text)
    if not blob:
        return None, 'no JSON object found in planner response'

    try:
        raw = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, f'spec is not valid JSON: {e}'
    if not isinstance(raw, dict):
        return None, f'spec must be a JSON object, got {type(raw).__name__}'

    def as_text(v):
        """Any JSON value flattened to a string; lists become space-joined."""
        if isinstance(v, (list, tuple)):
            return ' '.join(str(x) for x in v)
        return '' if v is None else str(v)

    def as_list(v):
        """A list of non-empty strings, accepting a comma-separated string too."""
        if v is None:
            return []
        if isinstance(v, str):
            return [p.strip() for p in v.split(',') if p.strip()]
        return [str(x).strip() for x in v if str(x).strip()]

    def as_map(v):
        """A string-to-string dict; a bare value is kept under 'note' rather than dropped."""
        if isinstance(v, dict):
            return {str(k): as_text(x) for k, x in v.items()}
        return {'note': as_text(v)} if v else {}

    spec = ExperimentSpec(
        hypothesis=as_text(raw.get('hypothesis')),
        mechanism=as_text(raw.get('mechanism')),
        evidence=as_map(raw.get('evidence')),
        target_metric=as_text(raw.get('target_metric')),
        proposed_change=as_text(raw.get('proposed_change')),
        expected_result=as_text(raw.get('expected_result')),
        falsification_condition=as_text(raw.get('falsification_condition')),
        risks=as_map(raw.get('risks')),
        implementation_scope=as_list(raw.get('implementation_scope')),
        tags=as_list(raw.get('tags')),
        candidates_considered=as_list(raw.get('candidates_considered')),
        reflection=as_text(raw.get('reflection')),
        prior_failure_attribution=as_text(raw.get('prior_failure_attribution')),
    )
    return spec, ''
