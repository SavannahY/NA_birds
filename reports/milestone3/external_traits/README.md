# External Trait Prompt Pipeline

This directory is for copyright-safe, compact external visual traits used as an
additive CLIP/SigLIP prompt ablation. The crawler stores extracted trait
phrases and provenance metadata, not raw field-guide prose. Raw API JSON is
cached under `reports/.cache/external_traits/` and should not be committed.

The committed artifact in this directory was built from Wikipedia text only
because it produced useful visual traits in the smoke test. GBIF matching is
implemented and useful for taxonomy/provenance, but its public descriptions
were mostly distribution text in the probe. EOL is implemented as an optional
source, but was slow from the local network.

## Build Locally

Smoke test without network:

```bash
python3 scripts/crawl_external_bird_traits.py --dry-run --limit 5
```

Limited crawl:

```bash
python3 scripts/crawl_external_bird_traits.py --limit 32
python3 scripts/match_external_traits_to_nabirds.py
python3 scripts/build_external_trait_prompts.py
```

Full crawl:

```bash
python3 scripts/crawl_external_bird_traits.py --sources wikipedia
python3 scripts/match_external_traits_to_nabirds.py
python3 scripts/build_external_trait_prompts.py
```

## Build On Modal

```bash
modal run scripts/modal_train_nabirds.py --task build-external-trait-prompts
```

For a smoke crawl on Modal:

```bash
modal run scripts/modal_train_nabirds.py --task build-external-trait-prompts --extra-args "--limit 32"
```

Then evaluate with cached embeddings:

```bash
modal run scripts/modal_train_nabirds.py --task precompute-vlm-text-features --prompts /data/nabirds_runs/reports/milestone3/external_traits/nabirds_external_trait_prompts.csv
modal run scripts/modal_train_nabirds.py --task eval-vlm-cached --feature-manifest /data/nabirds_runs/runs/milestone3/vlm_image_features/<manifest>.json --text-features /data/nabirds_runs/runs/vlm_text_features/<features>.pt
```

## Expected Artifacts

- `bird_traits.jsonl`: one row per base species with compact extracted traits,
  source status, source URLs, licenses, crawl timestamp, command, and API cache
  hashes.
- `external_trait_crawl_summary.json`: crawl counts and source status counts.
- `nabirds_external_trait_matches.csv`: 555 NABirds targets matched back to
  the external base-species traits.
- `external_trait_match_summary.json`: match counts.
- `nabirds_external_trait_prompts.csv`: evaluator-compatible prompt CSV with
  `target` and `prompt` columns.
- `external_trait_prompt_summary.json`: prompt row and class coverage counts.

Current committed counts:

- 404 base species rows crawled.
- 372 base species with at least one extracted visual trait.
- 517 of 555 NABirds target classes matched with external traits.
- 1,665 prompt rows, exactly 555 unique targets.

Current cached-evaluation results:

- CLIP ViT-B/32 + external trait prompts: top-1 40.03%, top-5 73.62%.
- SigLIP base + external trait prompts: top-1 51.61%, top-5 84.21%.
- Both runs saved text embeddings on the Modal volume under
  `/data/nabirds_runs/runs/milestone3_external_traits/vlm_text_features/` and
  reused existing saved image embeddings.

## Limitations

External API traits are base-species context. They do not reliably describe
NABirds sex, age, breeding, nonbreeding, morph, or regional variants. The
prompt builder therefore keeps the strong short variant label and adds external
traits only as an additional prompt template.

GBIF matches are conservative by default: exact species match, accepted
taxonomic status, and confidence at least 90. For NABirds common-name labels,
GBIF also has a vernacular-name search fallback. EOL first-result fallbacks are
marked ambiguous and ignored unless `--allow-eol-ambiguous` is set.
