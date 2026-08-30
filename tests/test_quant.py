"""
Unit tests for the quantitative pre-check (critic/quant.py).

Pure logic, no LLM: given a claim and the chunks it cites, does the
deterministic guard strike fabricated figures and pass real ones?

This is the module the v2 plan says never to cut, and the one where a bug
is quietest -- a weakened guard still reports success while checking less.
So the tests pin BOTH directions: what must be struck, and what must NOT be.

cited_chunks only needs the "text" key here -- build the smallest dict that
satisfies the function.

Optional new tool: `@pytest.mark.parametrize` lets one test function take a
list of (input, expected) pairs instead of copy-pasting near-identical
tests. Not required; separate functions often read better when they fail.
"""
from critic.quant import normalize, extract_numbers, check_quantities


def _chunks(*texts):
    """Minimal cited_chunks: check_quantities only reads the "text" key."""
    return [{"text": t} for t in texts]


# --- normalize: mechanical canonicalization only ---------------------------

def test_normalize_strips_whitespace_and_commas():
    """Formatting differences must not cause a strike."""
    assert normalize(" 92.7 % ") == "92.7%"
    assert normalize("1,200") == "1200"


def test_normalize_spells_out_percent():
    """"92.7 percent" and "92.7%" must normalize to the SAME string --
    otherwise a claim and its evidence miss each other over spelling."""
    assert normalize("92.7 percent") == normalize("92.7%")


def test_normalize_drops_trailing_zero_decimal_only():
    """The rule strips a meaningless .0 and never a real decimal."""
    assert normalize("41.0") == "41"
    assert normalize("92.7") == "92.7"


# --- extract_numbers: what counts as a number ------------------------------

def test_extracts_figures_with_units():
    """Percentages, bare counts, thousands separators and unit suffixes all
    come back normalized, in the order they appear."""
    text = "grounding 92.7%, 41 claims, 1,200 chunks, 3x faster, 10ms latency"
    assert extract_numbers(text) == ["92.7%", "41", "1200", "3x", "10ms"]


def test_multi_letter_units_are_not_invisible():
    """Regression, 2026-08-26: with only single letters in the unit list,
    "920MB" matched NOTHING -- `m` was consumed and `B` failed the trailing
    guard -- so a true claim ("920 MB layer") was struck against evidence
    reading "920MB". A number glued to a multi-character unit must still be
    found."""
    assert extract_numbers("verified 2.13.0+cpu, 920MB layer") == ["2.13", "920mb"]
    assert extract_numbers("bake took 210s and 186MB") == ["210s", "186mb"]


def test_spacing_around_a_unit_does_not_change_the_token():
    """The whole point of the fix: the claim spells it "920 MB", the commit
    message spells it "920MB", and the check must see one quantity."""
    assert extract_numbers("920 MB") == extract_numbers("920MB")


def test_unit_is_kept_so_different_quantities_stay_different():
    """Normalizing the unit AWAY would make a claim of 920 MB pass on evidence
    about 920ms -- same digits, different quantity, silent false pass."""
    assert extract_numbers("920mb") != extract_numbers("920ms")


def test_text_without_figures_yields_nothing():
    """The common, healthy case: prose with no digits must not raise."""
    assert extract_numbers("Refactored the planner service") == []


def test_sha_fragments_are_not_numbers():
    """"mealwise@8912ac4" must yield NOTHING. Both digit runs are
    glued to word characters; if either leaked, a commit sha in the evidence
    would make the haystack permissive enough to pass invented figures."""
    assert extract_numbers("mealwise@8912ac4") == []


def test_identifier_fragments_are_not_numbers():
    """"cross-encoder/ms-marco-MiniLM-L-6-v2" must yield NOTHING.
    Regression test -- this shipped as ['6'] and struck a true claim in the
    first make-star run."""
    assert extract_numbers("cross-encoder/ms-marco-MiniLM-L-6-v2") == []


def test_one_sided_hyphen_is_still_a_quantity():
    """"a 3-day plan" must still yield the 3, and a diff stat like
    "(+56/-32)" must still yield both numbers. The identifier rule keys on a
    hyphen on BOTH sides -- prove it didn't over-reach onto real figures."""
    assert extract_numbers("a 3-day plan") == ["3"]
    assert extract_numbers("(+56/-32)") == ["56","32"]


# --- check_quantities: the strike decision ---------------------------------

def test_claim_without_figures_passes():
    """Nothing to inflate; meaning is the LLM Critic's job, not this stage's."""
    ok, reason = check_quantities("Refactored the planner service",
                                  _chunks("unrelated evidence text"))
    assert ok is True
    assert reason == ""


def test_supported_figure_passes():
    """A figure present in the cited evidence survives."""
    ok, reason = check_quantities(
        "Grounding reached 92.7% on the eval set",
        _chunks("the run reports 92.7% of claims adequately supported"))
    assert ok is True
    assert reason == ""


def test_figure_may_come_from_any_cited_chunk():
    """The haystack is the union of all cited chunks, not each one alone."""
    ok, _ = check_quantities(
        "Indexed 41 chunks",
        _chunks("no digits in this one", "41 chunks were indexed"))
    assert ok is True


def test_fabricated_figure_is_struck():
    """a claim stating a figure absent from every cited chunk
    returns ok=False, AND the reason names that specific figure -- the reason
    reaches the artifact and the run trace, so "a number failed" is not
    good enough."""
    ok , reason = check_quantities(
        "Improved grounding by 40% across the eval set",
        _chunks("the run reports 92.7% of claims adequately supported")
    )
    assert ok is False
    assert "40%" in reason


def test_rounded_figure_is_struck():
    """claim "~93%" against evidence "92.7%" must STRIKE. This is
    the strict-exact-match policy decided 2026-07-19 -- rounding is a
    semantic judgment, so the Refiner rewrites to the exact figure instead.
    If this test ever fails, someone loosened the policy by accident."""
    ok , _ = check_quantities(
        "Grounding reached ~93% on the eval set",
        _chunks("the run reports 92.7% of claims adequately supported")
    )
    assert ok is False


def test_same_number_different_context_still_passes():
    """claim "41 chunks" against evidence "41 files" -> passes.
    Deliberate: this layer catches FABRICATED figures, the LLM Critic checks
    meaning. Pin it so the boundary between the two stages doesn't drift."""
    ok , _ = check_quantities(
        "41 chunks",
        _chunks("41 files were indexed")
    )
    assert ok is True
