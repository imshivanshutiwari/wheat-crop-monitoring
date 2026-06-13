# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
Transform the `wheat-crop-monitoring` codebase into a 10/10 production-tested portfolio project by resolving the GEE notebook loop bottleneck, verifying the correctness of all python modules locally using the test suite, and compiling the UAV GPR Research Report PDF to a premium format with all formatting errors fixed.

## Goals
1. **Optimize GEE loop in Notebook 03**: Refactor the sequential loop that makes 96 `getInfo()` calls (12 fortnights × 8 states) into a single batched `ee.FeatureCollection` map-reduction, retrieving all results in one `getInfo()` payload.
2. **Execute and Pass Test Suite**: Run and verify that all pytest unit tests for crop_model, phenology, unmixing, uncertainty, and yield_model pass successfully in the local workspace.
3. **Compile UAV GPR Research Report PDF**: Compile `generate_report.py` to generate the PDF research report at `outputs/UAV_GPR_Research_Report.pdf`, resolving all ReportLab formatting, styling, text-wrapping, margins, and layout overflow issues.
4. **Create Crop Analytics & Visualization Notebook**: Create a separate Jupyter notebook `09_crop_analytics_visualization.ipynb` that loads sample/output datasets and generates 20 distinct, publication-quality visualizations covering sown area, crop health (VHI), ML yields, and crop growth/weather ensembles.

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
- [ ] `generate_report.py` runs without error and generates `outputs/UAV_GPR_Research_Report.pdf`.
- [ ] All tables in the PDF wrap text properly inside cells (wrapped in `Paragraph` flowables) and do not overflow page margins.
- [ ] The output PDF layout matches premium design standards.
- [ ] Notebook `09_crop_analytics_visualization.ipynb` exists and runs successfully to produce 20 distinct crop analytics charts.
