# %% [markdown]
# # 04 - Data Relationships
#
# Day 1, step 4 - the last piece before any modelling. Notebook 01 deliberately merged
# nothing. This notebook decides, and *proves*, how the seven tables connect, then
# writes the conclusion to `docs/data_relationships.md`.
#
# The rule being enforced: a matching column name is not proof of a matching key.
# Every join below is verified by overlap and cardinality before it is accepted.

# %%
import pandas as pd

from app.utils.config import DOCS_DIR, PROCESSED_FILES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

emp = pd.read_csv(PROCESSED_FILES["attrition"])
eng = pd.read_csv(PROCESSED_FILES["engagement"])
occ = pd.read_csv(PROCESSED_FILES["occupation"])
req = pd.read_csv(PROCESSED_FILES["role_requirements"])
skills = pd.read_csv(PROCESSED_FILES["employee_skills"])

for n, d in [("attrition", emp), ("engagement", eng), ("occupation", occ),
             ("role_requirements", req), ("employee_skills", skills)]:
    print(f"{n:20s} {d.shape}")

# %% [markdown]
# ## The intended shape
#
# ```
# EMPLOYEE (employee_attrition)
#    |
#    +-- EmployeeID ----- ENGAGEMENT (hr_performance_engagement)   1:1
#    |
#    +-- EmployeeID ----- EMPLOYEE SKILLS                          1:many
#    |
#    +-- JobRole -------- ROLE REQUIREMENTS                        many:many via role
#                              |
#                              +-- ONET_SOC_Code -- OCCUPATION MASTER   many:1
# ```
#
# Employees connect to skills *through their role*, never directly. That indirection is
# the whole basis of the gap calculation: what the role requires, minus what the person
# has.

# %% [markdown]
# ## Helper: verify a candidate join

# %%
def verify_join(left: pd.DataFrame, right: pd.DataFrame, left_key: str, right_key: str,
                label: str) -> dict:
    lvals, rvals = set(left[left_key].dropna()), set(right[right_key].dropna())
    both = lvals & rvals
    l_card = "1" if left[left_key].is_unique else "many"
    r_card = "1" if right[right_key].is_unique else "many"
    return {
        "relationship": label,
        "key": left_key if left_key == right_key else f"{left_key} = {right_key}",
        "cardinality": f"{l_card}:{r_card}",
        "left_values": len(lvals),
        "right_values": len(rvals),
        "matched": len(both),
        "left_unmatched": len(lvals - rvals),
        "right_unmatched": len(rvals - lvals),
        "coverage_%": round(100 * len(both) / max(len(lvals), 1), 2),
    }


checks = [
    verify_join(emp, eng, "EmployeeID", "EmployeeID", "employee -> engagement"),
    verify_join(emp, skills, "EmployeeID", "EmployeeID", "employee -> employee_skills"),
    verify_join(emp, req, "JobRole", "JobRole", "employee -> role_requirements"),
    verify_join(req, occ, "ONET_SOC_Code", "ONET_SOC_Code", "role_requirements -> occupation"),
]
pd.DataFrame(checks)

# %% [markdown]
# ### Reading the results
#
# * **employee -> engagement** is `1:1` with 100% coverage. Same 1,470 IDs on both
#   sides, each appearing once. This join is safe.
# * **employee -> employee_skills** is `1:many` - one employee, several skill rows.
#   Coverage tells us whether anyone has no skills recorded at all.
# * **employee -> role_requirements** is `many:many` on `JobRole`. Both sides repeat, so
#   a naive `merge` here produces a row explosion; notebook 13 aggregates requirements
#   into a set per role *before* comparing.
# * **role_requirements -> occupation** is `many:1`, every SOC code resolving to exactly
#   one O*NET occupation.

# %% [markdown]
# ## The join that must not be done naively

# %%
naive = emp[["EmployeeID", "JobRole"]].merge(req, on="JobRole", how="left")
print(f"employees                 : {len(emp):,}")
print(f"role requirement rows     : {len(req):,}")
print(f"naive merged row count    : {len(naive):,}  <- {len(naive) // len(emp)}x explosion")
print("\nThis is correct for computing gaps, but it is NOT an employee-level table.")
print("Anything downstream that treats it as one will count every employee ~18 times.")

# %% [markdown]
# ## Verifying the 1:1 engagement join properly
#
# Equal row counts and matching ID sets are necessary but not sufficient - the columns
# have to agree about the same employee too.

# %%
merged = emp.merge(eng, on="EmployeeID", how="inner", suffixes=("", "_eng"))
print(f"inner join rows: {len(merged):,} (expected {len(emp):,})")

dept_agree = (merged["Department"] == merged["Department_eng"]).mean()
role_agree = (merged["JobRole"] == merged["JobRole_eng"]).mean()
print(f"Department agrees on {dept_agree:.1%} of rows")
print(f"JobRole    agrees on {role_agree:.1%} of rows")

# %% [markdown]
# Both agree completely. The two tables genuinely describe the same employees - the
# join key is confirmed, not merely assumed.

# %% [markdown]
# ## Role coverage

# %%
role_summary = (emp.groupby("JobRole")
                .agg(headcount=("EmployeeID", "count"),
                     attrition_rate=("AttritionFlag", lambda s: round(s.mean() * 100, 1)))
                .join(req.groupby("JobRole").size().rename("required_skills"))
                .join(req.groupby("JobRole")["ONET_SOC_Code"].first().rename("onet_soc"))
                .sort_values("headcount", ascending=False))
role_summary

# %%
missing_roles = set(emp["JobRole"]) - set(req["JobRole"])
print("Job roles with no skill requirements defined:", missing_roles or "none")
assert not missing_roles, "every job role must map to an O*NET occupation"

# %% [markdown]
# All nine IBM job roles map to an O*NET occupation, so no employee is left without a
# skill profile. The mapping and its rationale are in `docs/role_to_onet_mapping.csv`.

# %% [markdown]
# ## Employees with no skills on file

# %%
no_skills = set(emp["EmployeeID"]) - set(skills["EmployeeID"])
print(f"employees with zero recorded skills: {len(no_skills)}")
print("These must be treated as missing EVERY required skill, not as having no gap.")

# %% [markdown]
# ## Write the documented conclusion

# %%
rows = "\n".join(
    f"| {c['relationship']} | `{c['key']}` | {c['cardinality']} | {c['coverage_%']}% | "
    f"{c['left_unmatched']} |"
    for c in checks
)

doc = f"""# Data Relationships

Generated by `notebooks/04_data_relationships.ipynb`. Every relationship below was
verified by key overlap and cardinality, not assumed from column names.

## Entity diagram

```
EMPLOYEE (employee_attrition_processed)
   |
   +-- EmployeeID ----- ENGAGEMENT (engagement_processed)          1:1
   |
   +-- EmployeeID ----- EMPLOYEE SKILLS (employee_skills_processed) 1:many
   |
   +-- JobRole -------- ROLE REQUIREMENTS                           many:many
                             |
                             +-- ONET_SOC_Code -- OCCUPATION MASTER many:1
```

## Verified joins

| Relationship | Key | Cardinality | Coverage | Unmatched (left) |
|---|---|---|---|---|
{rows}

## Notes

* The IBM extract ships the key as `EmployeeNumber`; cleaning renames it to
  `EmployeeID` so it matches the name used across the platform. IDs are sparse
  (max {int(emp['EmployeeID'].max())} across {len(emp)} employees) - never treat the ID as a row index.
* `employee -> engagement` is genuinely 1:1: identical ID sets, and `Department` and
  `JobRole` agree on 100% of joined rows.
* `employee -> role_requirements` is many:many. Joining it directly multiplies the
  employee table roughly {len(req) // req['JobRole'].nunique()}x. Aggregate requirements to one set per role first.
* {len(no_skills)} employee(s) have no skills on file. They must be treated as missing every
  required skill.
* Engagement is a deterministic function of five columns the attrition model also uses.
  It is therefore **analytics-only** and is never fed back in as a model feature - doing
  so would be circular. See `scripts/build_derived_sources.py`.

## Join order for the intelligence table

1. `employee_attrition_processed` as the spine (one row per employee).
2. left join `engagement_processed` on `EmployeeID` (1:1).
3. left join per-employee skill gaps, pre-aggregated to one row per employee.
4. left join recommendations, pre-aggregated to one row per employee.

Steps 3 and 4 are aggregated *before* joining, which is what keeps the spine at one row
per employee.
"""

out = DOCS_DIR / "data_relationships.md"
out.write_text(doc, encoding="utf-8")
print(f"written: {out}")
print(doc[:900])
