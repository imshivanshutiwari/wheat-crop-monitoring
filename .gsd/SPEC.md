# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
Transform the `wheat-crop-monitoring` codebase into a 10/10 production-tested portfolio project by resolving the GEE notebook loop bottleneck and verifying the correctness of all python modules locally using the test suite.

## Goals
1. **Optimize GEE loop in Notebook 03**: Refactor the sequential loop that makes 96 `getInfo()` calls (12 fortnights × 8 states) into a single batched `ee.FeatureCollection` map-reduction, retrieving all results in one `getInfo()` payload.
2. **Execute and Pass Test Suite**: Run and verify that all pytest unit tests for crop_model, phenology, unmixing, uncertainty, and yield_model pass successfully in the local workspace.

## Non-Goals (Out of Scope)
- Setting up active GEE authentication or mock GEE servers for unit testing GEE functions (we rely on the lazy-import offline design).
- Replacing synthetic fallback datasets with real proprietary IMD/AWiFS datasets in this repository.

## Users
- Model reviewers, crop monitoring researchers, and portfolio evaluators.

## Constraints
- Windows PowerShell local environment.
- Earth Engine operations require GEE authentication to run notebooks, but local python tests must run entirely offline without GEE credentials.

## Success Criteria
- [ ] Notebook `03_vhi_crop_health_monitoring.ipynb` contains the optimized batched GEE code.
- [ ] Local tests run via `python -m pytest` pass successfully.
