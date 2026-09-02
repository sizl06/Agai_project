"""Canonicalise skill names.

The build notes flag this directly: 'AWS' vs 'Amazon Web Services' vs 'AWS Cloud'
are one skill, and if they stay distinct the gap engine counts the same shortage
three times and recommends training an employee already has.

Two layers:
  1. A generic cleanup - whitespace, known source typos, and O*NET's habit of
     appending " software" to product names ("SAP software" -> "SAP").
  2. An explicit alias table for names that generic rules cannot join, including
     the abbreviations a real HR skills export would contain.

Matching is case- and punctuation-insensitive, so "aws", "A.W.S." and "AWS " all
resolve to the same canonical name.
"""
from __future__ import annotations

import re

# Typos present in the upstream O*NET extract.
_TYPO_FIXES = {
    "micosoft": "microsoft",
    "microsft": "microsoft",
}

# Suffixes O*NET appends to product names. Order matters: longest first.
_STRIPPABLE_SUFFIXES = (
    " software",
    " systems",
    " system",
)

# Canonical name keyed by normalised lookup key. Everything on the right is the
# single spelling the rest of the platform uses.
_ALIASES = {
    # --- cloud ---
    "aws": "Amazon Web Services (AWS)",
    "aws cloud": "Amazon Web Services (AWS)",
    "amazon web services": "Amazon Web Services (AWS)",
    "amazon web services aws": "Amazon Web Services (AWS)",
    "azure": "Microsoft Azure",
    "ms azure": "Microsoft Azure",
    "microsoft azure": "Microsoft Azure",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    # --- analytics / languages ---
    "the mathworks matlab": "MATLAB",
    "matlab": "MATLAB",
    "python3": "Python",
    "python 3": "Python",
    "r": "R",
    "r language": "R",
    "sas": "SAS",
    "ibm spss statistics": "IBM SPSS Statistics",
    "spss": "IBM SPSS Statistics",
    "apache hadoop": "Apache Hadoop",
    "hadoop": "Apache Hadoop",
    "sql": "SQL",
    "structured query language": "SQL",
    # --- office ---
    "microsoft office": "Microsoft Office",
    "ms office": "Microsoft Office",
    "microsoft excel": "Microsoft Excel",
    "ms excel": "Microsoft Excel",
    "excel": "Microsoft Excel",
    "microsoft word": "Microsoft Word",
    "ms word": "Microsoft Word",
    "microsoft powerpoint": "Microsoft PowerPoint",
    "powerpoint": "Microsoft PowerPoint",
    "microsoft outlook": "Microsoft Outlook",
    "outlook": "Microsoft Outlook",
    "google workspace": "Google Workspace",
    "g suite": "Google Workspace",
    "google docs": "Google Docs",
    # --- enterprise ---
    "sap": "SAP",
    "salesforce": "Salesforce",
    "salesforce crm": "Salesforce",
    "workday": "Workday",
    "hubspot": "HubSpot",
    "oracle hris": "Oracle HRIS",
    "applicant tracking": "Applicant Tracking System (ATS)",
    "ats": "Applicant Tracking System (ATS)",
    "electronic medical record emr": "Electronic Medical Record (EMR)",
    "emr": "Electronic Medical Record (EMR)",
    "meditech": "MEDITECH",
    # --- collaboration / dev ---
    "atlassian jira": "Atlassian JIRA",
    "jira": "Atlassian JIRA",
    "atlassian confluence": "Atlassian Confluence",
    "confluence": "Atlassian Confluence",
    "cisco webex": "Cisco Webex",
    "webex": "Cisco Webex",
    "microsoft sharepoint": "Microsoft SharePoint",
    "sharepoint": "Microsoft SharePoint",
    "linux": "Linux",
    "apple macos": "Apple macOS",
    "macos": "Apple macOS",
    "email": "Email",
    "electronic mail": "Email",
    # --- design ---
    "adobe creative cloud": "Adobe Creative Cloud",
    "adobe photoshop": "Adobe Photoshop",
    "photoshop": "Adobe Photoshop",
    "adobe illustrator": "Adobe Illustrator",
    "adobe indesign": "Adobe InDesign",
    "adobe acrobat": "Adobe Acrobat",
    "adobe after effects": "Adobe After Effects",
    "autodesk autocad": "Autodesk AutoCAD",
    "autocad": "Autodesk AutoCAD",
    "esri arcgis": "ESRI ArcGIS",
    "arcgis": "ESRI ArcGIS",
    "google analytics": "Google Analytics",
    "microsoft access": "Microsoft Access",
    "microsoft project": "Microsoft Project",
    "microsoft visio": "Microsoft Visio",
}


def _lookup_key(name: str) -> str:
    """Reduce a name to a comparison key: lowercase, no punctuation, no suffix."""
    key = str(name).strip().lower()
    key = re.sub(r"[.\-_/(),]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()

    for wrong, right in _TYPO_FIXES.items():
        key = key.replace(wrong, right)

    changed = True
    while changed:
        changed = False
        for suffix in _STRIPPABLE_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix) + 1:
                key = key[: -len(suffix)].strip()
                changed = True
    return key


def normalize_skill(name: str) -> str:
    """Return the canonical spelling of a single skill name."""
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""

    key = _lookup_key(raw)
    if key in _ALIASES:
        return _ALIASES[key]

    # Dotted acronyms ("A.W.S.", "G.C.P.") lose their dots to spaces above, leaving
    # "a w s". Retry with separators stripped so they reach the same alias. Tried
    # second so a genuine multi-word name is never mangled into a false match.
    collapsed = key.replace(" ", "")
    if collapsed in _ALIASES:
        return _ALIASES[collapsed]

    # No alias: keep the cleaned-up form, preserving the original capitalisation
    # of each surviving word so "ESRI ArcGIS" does not become "Esri Arcgis".
    cleaned = re.sub(r"\s+", " ", raw)
    for suffix in _STRIPPABLE_SUFFIXES:
        if cleaned.lower().endswith(suffix) and len(cleaned) > len(suffix) + 1:
            cleaned = cleaned[: -len(suffix)].strip()
    for wrong, right in _TYPO_FIXES.items():
        cleaned = re.sub(wrong, right.capitalize(), cleaned, flags=re.IGNORECASE)
    return cleaned


def normalize_series(values) -> list[str]:
    """Vectorised convenience wrapper for a pandas Series or any iterable."""
    return [normalize_skill(v) for v in values]


def normalization_audit(values) -> dict[str, list[str]]:
    """Map each canonical name to the raw spellings that collapsed into it.

    Used in the cleaning notebook to show what the normaliser actually merged,
    rather than asserting that it worked.
    """
    audit: dict[str, set[str]] = {}
    for raw in values:
        canon = normalize_skill(raw)
        audit.setdefault(canon, set()).add(str(raw).strip())
    return {k: sorted(v) for k, v in sorted(audit.items()) if len(v) > 1}
