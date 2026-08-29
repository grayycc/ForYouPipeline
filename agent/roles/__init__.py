"""One module per LLM role. The orchestrator calls them in a fixed order and never builds a
prompt itself, so each role's behaviour is readable in a single file."""
