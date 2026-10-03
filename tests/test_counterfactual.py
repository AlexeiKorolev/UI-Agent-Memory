import random

from PIL import Image, ImageDraw, ImageFont

from src.counterfactual import FONT, choose_alternative, edit_crop, edit_text, perturb_digits, type_class
from src.ocr import best_window_sim, ocr_image


def test_type_class():
    assert type_class("9298916954") == "digits"
    assert type_class("https://docs.google.com/x") == "url"
    assert type_class("Grilled Chicken Salad") == "alpha"


def test_perturb_digits_keeps_format():
    rng = random.Random(0)
    s = "11400 Roachester Ave, (609) 555-0123"
    alt = perturb_digits(s, rng)
    assert len(alt) == len(s)
    assert all((a.isdigit() == b.isdigit()) for a, b in zip(s, alt))
    assert all(a != b for a, b in zip(s, alt) if a.isdigit())
    assert alt[0] != "0"


def test_choose_alternative_same_type_and_forbidden():
    rng = random.Random(0)
    pool = ["Blue Lagoon Cafe", "Grilled Fish Tacos", "Mango Sticky Rice", "https://a.com/x"]
    alt, how = choose_alternative("Grilled Chicken Salad", pool, rng, forbid_texts=("Mango Sticky Rice is great",))
    assert how == "pool_swap" and alt in ("Blue Lagoon Cafe",)  # 3 words, not similar, not forbidden
    alt, how = choose_alternative("9298916954", pool, rng)
    assert how == "digit_perturb" and len(alt) == 10 and alt.isdigit()


def test_edit_text_exact_and_fuzzy():
    s, k = edit_text("Found phone 929-891-6954 for Joe's Pizza.", "929-891-6954", "111-222-3333")
    assert "111-222-3333" in s and "929" not in s and k >= 1
    s, k = edit_text("Nobel winners: Arthur Ashkin, Gerard Mourou.", "Arthur Ashkn", "Bella Torres")
    assert "Bella Torres" in s and "Ashkin" not in s


def test_edit_crop_replaces_rendered_text():
    im = Image.new("RGB", (900, 160), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((20, 50), "Total: 4821 Maple Street", fill=(20, 20, 20), font=ImageFont.truetype(FONT, 48))
    assert best_window_sim("4821 Maple Street", ocr_image(im)["words"])[0] >= 0.8
    out, found = edit_crop(im, "4821 Maple Street", "7390 Cedar Road")
    assert found
    words = ocr_image(out)["words"]
    assert best_window_sim("4821 Maple Street", words)[0] < 0.8
    assert best_window_sim("7390 Cedar Road", words)[0] >= 0.8
    assert best_window_sim("Total", words)[0] >= 0.8  # untouched context survives
