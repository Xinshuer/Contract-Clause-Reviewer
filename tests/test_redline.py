from clausecheck.redline import locate_anchor, make_diff, split_sentences

CLAUSE = (
    "All Fees are non-refundable under any circumstances, including where Customer terminates "
    "this Agreement for Provider's breach. Overdue amounts accrue interest at three percent (3%) per month."
)


def test_exact_anchor():
    span = locate_anchor(CLAUSE, "Overdue amounts accrue interest at three percent (3%) per month.")
    assert span and CLAUSE[span[0]:span[1]].startswith("Overdue")


def test_whitespace_insensitive_anchor():
    span = locate_anchor(CLAUSE, "All Fees are non-refundable\nunder any   circumstances")
    assert span == (0, len("All Fees are non-refundable under any circumstances"))


def test_fuzzy_anchor_within_sentence():
    # model dropped a comma - still >= 95 similar to the real sentence
    span = locate_anchor(CLAUSE, "All Fees are non-refundable under any circumstances including where Customer terminates this Agreement for Provider's breach.")
    assert span is not None
    assert CLAUSE[span[0]:span[1]].endswith("breach.")


def test_missing_anchor_is_none_not_a_guess():
    assert locate_anchor(CLAUSE, "Provider shall refund all Fees on request.") is None
    assert locate_anchor(CLAUSE, "") is None


def test_diff_and_sentences():
    s = split_sentences(CLAUSE)
    assert len(s) == 2
    span = locate_anchor(CLAUSE, s[1])
    diff = make_diff(CLAUSE, span[0], span[1], "Overdue amounts accrue statutory interest.", "3.2")
    assert "-" in diff and "+" in diff and "statutory interest" in diff
