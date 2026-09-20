from pathlib import Path

from clausecheck.split import load_contract, split_clauses

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_contract.txt"


def test_sample_contract_splits_into_numbered_clauses():
    contract = load_contract(SAMPLE)
    ids = [c.clause_id for c in contract.clauses]
    assert "8.1" in ids and "12" in ids and "15.2" in ids
    assert ids == sorted(ids, key=lambda s: tuple(int(x) for x in s.split(".")))


def test_cross_references_are_extracted():
    contract = load_contract(SAMPLE)
    assert "12" in contract.get("8.1").refs
    assert "4.2" in contract.get("11.2").refs
    assert "9.1" in contract.get("12").refs and "8" in contract.get("12").refs


def test_section_titles_have_children_text():
    contract = load_contract(SAMPLE)
    sec = contract.get("8")
    assert sec.is_section_title
    assert "aggregate liability" in contract.full_text_of("8")
    assert contract.full_text_of("99") is None


def test_breadcrumb_path_and_offsets():
    contract = load_contract(SAMPLE)
    c = contract.get("8.1")
    assert c.section_path.startswith("8 Limitation of Liability > 8.1")
    assert c.text.strip() in contract.text[c.char_start:c.char_end]


def test_clause_spanning_a_page_break_stays_whole():
    # Simulate a page break inside clause 2: the parser joins pages before splitting.
    page1 = "1. Scope\nThis clause is short.\n2. Payment\nFees are payable within thirty (30) days\n"
    page2 = "of the invoice date. Late amounts accrue interest.\n3. Term\nOne year.\n"
    clauses = split_clauses(page1 + page2)
    pay = next(c for c in clauses if c.clause_id == "2")
    assert "thirty (30) days\nof the invoice date" in pay.text
    assert len(clauses) == 3
