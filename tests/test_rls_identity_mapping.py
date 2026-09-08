"""Row-level security has to compare an identity with an identity.

The Sales Rep role filtered `dim_customer[rep] = USERPRINCIPALNAME()`. `rep`
holds display names — "D. Tremblay" — and a signed-in user's UPN is an email
address. Those are never equal, so the role returned an empty report to every
real user: the failure that reads as "the data didn't load" rather than as a
broken filter, and the one a demo with impersonation-by-role never shows,
because impersonating a *role* skips the identity comparison entirely.

The fix is a table that says which identity maps to which rep. These tests pin
that it exists, that it is joined into the predicate, and that the predicate no
longer compares a name to a UPN.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "powerbi/pbip/CustomerProductAnalytics.SemanticModel/definition"
ROLE = MODEL / "roles" / "Sales Rep.tmdl"
MAPPING_TABLE = MODEL / "tables" / "security_mapping.tmdl"
MAPPING_CSV = ROOT / "output" / "security_mapping.csv"


def _role_text() -> str:
    assert ROLE.is_file(), "the Sales Rep role is gone"
    return " ".join(ROLE.read_text(encoding="utf-8").split())


def test_the_predicate_does_not_compare_a_name_to_a_upn():
    role = _role_text()
    assert "dim_customer[rep] = USERPRINCIPALNAME()" not in role, (
        "the role compares a rep's display name against a sign-in identity; "
        "no real user will ever match, so the report comes back empty"
    )


def test_the_predicate_resolves_the_identity_through_the_mapping():
    role = _role_text()
    assert "USERPRINCIPALNAME()" in role, "the role no longer scopes by the signed-in user"
    assert "security_mapping[upn]" in role, (
        "the identity is not looked up in the mapping table"
    )
    assert "security_mapping[rep]" in role, (
        "the mapping's rep column is not what dim_customer is filtered to"
    )


def test_the_mapping_table_is_defined_and_registered():
    assert MAPPING_TABLE.is_file(), "security_mapping.tmdl is missing"
    text = MAPPING_TABLE.read_text(encoding="utf-8")
    for column in ("upn", "rep"):
        assert re.search(rf"column {column}\b", text), f"security_mapping has no {column} column"
    model = (MODEL / "model.tmdl").read_text(encoding="utf-8")
    assert "ref table security_mapping" in model, (
        "the table exists but the model does not load it, so the role's "
        "predicate references a table Power BI cannot resolve"
    )


def test_the_mapping_covers_every_rep_and_holds_real_upns():
    if not MAPPING_CSV.is_file():
        pytest.skip("run analytics/customer_analytics.py to build the mapping")
    rows = list(csv.DictReader(MAPPING_CSV.open(encoding="utf-8")))
    assert rows, "the mapping is empty, which scopes every rep to nothing"
    assert {"upn", "rep"} <= set(rows[0]), "the mapping is missing a column"

    customers = ROOT / "output" / "customer_analytics.csv"
    if customers.is_file():
        reps = {r["rep"] for r in csv.DictReader(customers.open(encoding="utf-8")) if r.get("rep")}
        mapped = {r["rep"] for r in rows}
        missing = reps - mapped
        assert not missing, f"reps with no sign-in identity, so they see nothing: {sorted(missing)}"

    for r in rows:
        assert "@" in r["upn"], f"{r['upn']!r} is not a sign-in identity"
        assert r["upn"] == r["upn"].lower(), "UPNs should be compared in one case"
