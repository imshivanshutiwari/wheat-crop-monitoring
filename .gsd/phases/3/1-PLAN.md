---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Compile UAV GPR PDF and Fix Formatting Errors

## Objective
Refactor `generate_report.py` to fix ReportLab layout, formatting, text wrapping, and margin issues, then compile it to generate a beautiful, premium PDF at `outputs/UAV_GPR_Research_Report.pdf`.

## Context
- .gsd/SPEC.md
- generate_report.py

## Tasks

<task type="auto">
  <name>Verify environment dependencies and adjust output path</name>
  <files>generate_report.py</files>
  <action>
    - Verify if ReportLab is installed in the active virtual environment (`venv`). If not, install it.
    - Update the `OUTPUT` path in `generate_report.py` to point to a local directory in the workspace: `outputs/UAV_GPR_Research_Report.pdf` instead of the Linux path `/mnt/user-data/outputs/UAV_GPR_Research_Report.pdf`.
    - Ensure the `outputs` directory is created dynamically before compiling.
  </action>
  <verify>python -c "import reportlab; print(reportlab.__version__)"</verify>
  <done>ReportLab is installed, and the output path is set correctly in the workspace.</done>
</task>

<task type="auto">
  <name>Refactor tables, styles, text wrapping, and margins</name>
  <files>generate_report.py</files>
  <action>
    - Change margins to 1.5*cm all around (`leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm`) to maximize printable width to 18.0 cm on A4.
    - Create a custom cell paragraph style (e.g. `cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor('#333333'))`) and a header cell style (e.g. `header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontSize=8, leading=10, fontName='Helvetica-Bold', textColor=colors.white)`).
    - Wrap all table cell contents (except numbers/simple strings that won't wrap/overflow) in `Paragraph` flowables using these cell/header styles. This ensures that newlines (`\n`) are rendered correctly and long text wraps dynamically instead of clipping or overflowing.
    - Adjust column widths for all tables so they sum up to exactly 18.0 cm (printable page width):
      - `sys_table`: `[3.2*cm, 2.0*cm, 0.8*cm, 1.6*cm, 1.6*cm, 1.8*cm, 1.0*cm, 6.0*cm]` (Sum = 18.0 cm)
      - `ds_table`: `[3.2*cm, 2.2*cm, 1.8*cm, 3.5*cm, 1.3*cm, 6.0*cm]` (Sum = 18.0 cm)
      - `ai_table`: `[3.0*cm, 2.0*cm, 2.2*cm, 2.2*cm, 4.3*cm, 4.3*cm]` (Sum = 18.0 cm)
      - `feas_table`: `[4.0*cm, 6.0*cm, 8.0*cm]` (Sum = 18.0 cm)
      - `swot_table`: `[9.0*cm, 9.0*cm]` (Sum = 18.0 cm)
  </action>
  <verify>python generate_report.py</verify>
  <done>All table data is wrapped in Paragraphs, column widths sum to 18 cm, and the script compiles successfully.</done>
</task>

<task type="auto">
  <name>Final verification of PDF generation</name>
  <files>outputs/UAV_GPR_Research_Report.pdf</files>
  <action>
    - Run `generate_report.py` to compile the PDF.
    - Verify that the output PDF file `outputs/UAV_GPR_Research_Report.pdf` exists and is non-empty.
  </action>
  <verify>powershell "Test-Path outputs/UAV_GPR_Research_Report.pdf"</verify>
  <done>The PDF file is compiled successfully and resides in the output directory.</done>
</task>

<task type="auto">
  <name>Create Crop Analytics and Visualization Notebook</name>
  <files>notebooks/09_crop_analytics_visualization.ipynb</files>
  <action>
    - Create a Jupyter notebook `notebooks/09_crop_analytics_visualization.ipynb`.
    - Implement python cells that load:
      - `data/sample/district_yield_history.csv`
      - `data/sample/ministry_ground_truth.csv`
      - Any output CSV files from previous notebooks or generate consistent mock data if they aren't generated.
    - Generate 20 distinct publication-quality visualizations using matplotlib and seaborn.
    - Save the notebook with all outputs rendered.
  </action>
  <verify>powershell "Test-Path notebooks/09_crop_analytics_visualization.ipynb"</verify>
  <done>Notebook `09_crop_analytics_visualization.ipynb` is created and runs without error, rendering 20 beautiful graphs.</done>
</task>

## Success Criteria
- [ ] `generate_report.py` runs without error and generates `outputs/UAV_GPR_Research_Report.pdf`.
- [ ] Table cells are fully wrapped with no text clippings or page-width overflows.
- [ ] Output PDF meets high-quality design standards.
- [ ] Notebook `notebooks/09_crop_analytics_visualization.ipynb` is created and contains exactly 20 distinct graphs and visualizations, all rendered correctly.
