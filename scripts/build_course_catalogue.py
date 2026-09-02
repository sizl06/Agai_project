"""Build the training catalogue the recommendation engine maps gaps onto.

This catalogue is SYNTHETIC. Course titles, providers and durations are plausible
placeholders standing in for whatever LMS or vendor feed a real deployment would
plug in. What matters architecturally is the shape: every course carries a
`covers_skills` tag list, which is how real LMS catalogues (Coursera for Business,
Udemy Business, LinkedIn Learning) actually expose their content.

The descriptions are deliberately written *without* always repeating the skill name,
because that is what makes the semantic-matching step in notebook 15 a real test
rather than a keyword lookup that trivially succeeds.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.config import EXTERNAL_DIR  # noqa: E402

# (title, provider, level, hours, description, [skills covered])
COURSES = [
    # --- data & cloud ---
    ("Python for Data Analysis", "DataCamp", "Beginner", 20,
     "Write clean scripts, wrangle tabular data with pandas, and automate repetitive "
     "reporting work end to end.", ["Python"]),
    ("Statistical Programming in R", "Coursera", "Intermediate", 24,
     "Model and visualise experimental results using the tidyverse, from raw readings "
     "to a publication-ready figure.", ["R"]),
    ("Applied Statistics with SPSS", "IBM SkillsBuild", "Beginner", 16,
     "Run hypothesis tests, regression and survival analysis through a point-and-click "
     "workflow aimed at researchers.", ["IBM SPSS Statistics"]),
    ("SAS Programming Essentials", "SAS Institute", "Beginner", 18,
     "Manipulate large clinical and operational datasets using DATA steps and PROC "
     "procedures.", ["SAS"]),
    ("Numerical Computing and Simulation", "MathWorks", "Intermediate", 22,
     "Build and validate numerical models, solve systems of equations and visualise "
     "simulation output.", ["MATLAB", "Mathematics"]),
    ("Deploying and Monitoring Cloud Workloads", "A Cloud Guru", "Intermediate", 30,
     "Provision compute and storage, wire up managed services, and keep production "
     "systems observable and costed.", ["Amazon Web Services (AWS)", "Microsoft Azure"]),
    ("Distributed Data Processing at Scale", "Cloudera", "Advanced", 28,
     "Process datasets far larger than memory across a cluster, covering storage "
     "formats, partitioning and job tuning.", ["Apache Hadoop"]),
    ("Linux Command Line and Shell Scripting", "Linux Foundation", "Beginner", 14,
     "Navigate a server confidently, chain utilities together and automate routine "
     "administration.", ["Linux"]),
    ("Relational Databases and Query Design", "Coursera", "Beginner", 18,
     "Design normalised tables, write efficient queries and reason about indexes and "
     "joins.", ["Microsoft Access", "Oracle HRIS"]),

    # --- office & collaboration ---
    ("Advanced Spreadsheet Modelling", "Microsoft Learn", "Intermediate", 12,
     "Build maintainable financial and operational models using lookup functions, "
     "pivot tables and what-if analysis.", ["Microsoft Excel"]),
    ("Business Documents That Get Read", "LinkedIn Learning", "Beginner", 8,
     "Structure long-form documents with styles, templates and review workflows so "
     "they survive collaborative editing.", ["Microsoft Word", "Google Docs"]),
    ("Presenting Data to Decision Makers", "LinkedIn Learning", "Beginner", 10,
     "Turn analysis into a narrative a busy executive can follow, with slide design "
     "and delivery practice.", ["Microsoft PowerPoint", "Speaking"]),
    ("Productivity Suite Fundamentals", "Microsoft Learn", "Beginner", 10,
     "Get fluent across the everyday toolset - documents, spreadsheets, slides, mail "
     "and shared drives.", ["Microsoft Office", "Google Workspace", "Microsoft Outlook", "Email"]),
    ("Team Knowledge Bases and Intranets", "Atlassian", "Beginner", 8,
     "Organise shared documentation so information can actually be found six months "
     "later.", ["Microsoft SharePoint", "Atlassian Confluence"]),
    ("Running Effective Remote Meetings", "LinkedIn Learning", "Beginner", 6,
     "Facilitate distributed sessions that stay on track, including recording, "
     "breakouts and follow-up.", ["Cisco Webex", "Active Listening"]),
    ("Agile Delivery with Issue Tracking", "Atlassian", "Intermediate", 14,
     "Plan sprints, manage backlogs and report progress with boards and workflows.",
     ["Atlassian JIRA", "Microsoft Project"]),
    ("Project Planning and Scheduling", "PMI", "Intermediate", 20,
     "Build realistic schedules, manage dependencies and critical paths, and track "
     "against a baseline.", ["Microsoft Project", "Monitoring"]),
    ("Process Mapping and Diagramming", "Microsoft Learn", "Beginner", 8,
     "Document workflows and system architectures clearly enough for someone else to "
     "act on them.", ["Microsoft Visio"]),
    ("Working with PDFs and Document Workflows", "Adobe", "Beginner", 6,
     "Create, combine, annotate and secure portable documents, including forms and "
     "signatures.", ["Adobe Acrobat"]),

    # --- design & media ---
    ("Digital Image Editing Foundations", "Adobe", "Beginner", 16,
     "Retouch, composite and prepare imagery for print and screen with non-destructive "
     "techniques.", ["Adobe Photoshop", "Adobe Creative Cloud"]),
    ("Vector Graphics and Brand Assets", "Adobe", "Intermediate", 14,
     "Produce scalable logos, icons and illustrations that hold up at any size.",
     ["Adobe Illustrator"]),
    ("Layout and Publication Design", "Adobe", "Intermediate", 14,
     "Set type and lay out multi-page publications with grids, styles and print-ready "
     "export.", ["Adobe InDesign"]),
    ("Motion Graphics and Video Post-Production", "Adobe", "Advanced", 20,
     "Animate, composite and grade footage for marketing and training material.",
     ["Adobe After Effects"]),
    ("Technical Drawing and 3D Modelling", "Autodesk", "Intermediate", 26,
     "Produce dimensioned engineering drawings and parametric models for manufacture.",
     ["Autodesk AutoCAD"]),
    ("Spatial Analysis and Mapping", "ESRI", "Intermediate", 22,
     "Analyse location data, build map layers and communicate geographic patterns.",
     ["ESRI ArcGIS"]),

    # --- sales, marketing, HR, health ---
    ("Managing the Customer Pipeline", "Trailhead", "Beginner", 16,
     "Track opportunities from first contact to close, keep records clean and report "
     "on the funnel.", ["Salesforce", "HubSpot"]),
    ("Inbound Marketing and Web Analytics", "HubSpot Academy", "Beginner", 12,
     "Attract and measure an audience, interpreting traffic and conversion data to "
     "guide campaigns.", ["HubSpot", "Google Analytics"]),
    ("Measuring Digital Engagement", "Google", "Beginner", 10,
     "Set up tracking, build reports and interpret behavioural metrics without "
     "over-reading noise.", ["Google Analytics"]),
    ("Enterprise Resource Planning Essentials", "SAP", "Intermediate", 24,
     "Understand how finance, procurement and logistics modules connect across a "
     "single business system.", ["SAP"]),
    ("Modern HR Information Systems", "Workday", "Beginner", 14,
     "Administer employee records, absence and compensation in a cloud HR platform.",
     ["Workday", "Oracle HRIS"]),
    ("Recruiting Operations and Candidate Tracking", "LinkedIn Learning", "Beginner", 10,
     "Run a structured hiring pipeline, from requisition through offer, without losing "
     "candidates in the process.", ["Applicant Tracking System (ATS)"]),
    ("Clinical Records and Patient Data Systems", "HIMSS", "Intermediate", 18,
     "Work accurately and compliantly with patient information, including coding and "
     "privacy obligations.", ["Electronic Medical Record (EMR)", "MEDITECH"]),

    # --- essential / cognitive skills ---
    ("Structured Problem Solving", "Coursera", "Intermediate", 12,
     "Break ambiguous problems into testable parts, weigh evidence and avoid common "
     "reasoning traps.", ["Critical Thinking"]),
    ("Learning How to Learn", "Coursera", "Beginner", 8,
     "Evidence-based study technique - spaced repetition, retrieval practice and "
     "deliberate practice applied to work skills.", ["Active Learning", "Learning Strategies"]),
    ("Business Writing for Clarity", "LinkedIn Learning", "Beginner", 8,
     "Write documents people act on: structure first, plain language, and editing "
     "down to what matters.", ["Writing", "Reading Comprehension"]),
    ("Listening and Influence at Work", "LinkedIn Learning", "Beginner", 6,
     "Draw out what colleagues actually mean, and handle difficult conversations "
     "without escalating them.", ["Active Listening", "Speaking"]),
    ("Quantitative Reasoning for Professionals", "Khan Academy", "Beginner", 16,
     "Refresh the arithmetic, algebra and statistics that underpin everyday business "
     "decisions.", ["Mathematics"]),
    ("Scientific Method in Practice", "edX", "Intermediate", 14,
     "Design experiments, control for confounders and interpret results honestly.",
     ["Science"]),
    ("Performance Tracking and Review", "LinkedIn Learning", "Beginner", 6,
     "Set measurable objectives and monitor progress against them without "
     "micromanaging.", ["Monitoring"]),
    ("Operating Systems for End Users", "Apple", "Beginner", 6,
     "Work efficiently on a desktop platform - files, permissions, backup and "
     "troubleshooting.", ["Apple macOS"]),
]


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EXTERNAL_DIR / "course_catalogue.csv"

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["CourseID", "Title", "Provider", "Level", "DurationHours",
                         "Description", "CoversSkills"])
        for i, (title, provider, level, hours, description, skills) in enumerate(COURSES, 1):
            writer.writerow([f"C{i:03d}", title, provider, level, hours,
                             description, "|".join(skills)])

    covered = {s for *_, skills in COURSES for s in skills}
    print(f"{out.name}: {len(COURSES)} courses covering {len(covered)} distinct skills")


if __name__ == "__main__":
    main()
