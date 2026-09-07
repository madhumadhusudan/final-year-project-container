"""Deterministic dummy image generation for privacy validation.

No payload or rendered identifier belongs to a real person or account.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .metrics import Box


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures"


@dataclass(frozen=True)
class LabeledImage:
    name: str
    pixels: np.ndarray
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class CompositeScene:
    pixels: np.ndarray
    boxes: dict[str, tuple[Box, ...]]


def load_fixture(name: str) -> np.ndarray:
    pixels = cv2.imread(str(FIXTURE_DIRECTORY / name), cv2.IMREAD_COLOR)
    if pixels is None:
        raise RuntimeError(f"Synthetic fixture is missing: {name}")
    return pixels


def qr_image(payload: str, scale: int = 7, border: int = 40) -> np.ndarray:
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    scaled = cv2.resize(encoded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((scaled.shape[0] + border * 2, scaled.shape[1] + border * 2, 3), 255, np.uint8)
    canvas[border:border + scaled.shape[0], border:border + scaled.shape[1]] = cv2.cvtColor(
        scaled, cv2.COLOR_GRAY2BGR
    )
    return canvas


def rotate_bound(image: np.ndarray, angle: float, fill: int = 255) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    target_width = int(height * sine + width * cosine)
    target_height = int(height * cosine + width * sine)
    matrix[0, 2] += target_width / 2 - center[0]
    matrix[1, 2] += target_height / 2 - center[1]
    return cv2.warpAffine(
        image, matrix, (target_width, target_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(fill, fill, fill),
    )


def ean13(value: str = "5901234123457", module: int = 4) -> np.ndarray:
    left = {
        "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101", "4": "0100011",
        "5": "0110001", "6": "0101111", "7": "0111011", "8": "0110111", "9": "0001011",
    }
    alternate = {
        "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001", "4": "0011101",
        "5": "0111001", "6": "0000101", "7": "0010001", "8": "0001001", "9": "0010111",
    }
    right = {digit: pattern.translate(str.maketrans("01", "10")) for digit, pattern in left.items()}
    parity = {
        "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL", "4": "LGLLGG",
        "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG", "8": "LGLGGL", "9": "LGGLGL",
    }
    check = (10 - sum((3 if i % 2 else 1) * int(digit) for i, digit in enumerate(value[:12])) % 10) % 10
    if len(value) != 13 or int(value[-1]) != check:
        raise ValueError("EAN-13 dummy value must contain a valid check digit")
    left_bits = "".join((left if kind == "L" else alternate)[digit] for digit, kind in zip(value[1:7], parity[value[0]]))
    bits = "101" + left_bits + "01010" + "".join(right[digit] for digit in value[7:]) + "101"
    quiet = 14
    image = np.full((210, (len(bits) + quiet * 2) * module, 3), 255, np.uint8)
    for index, bit in enumerate(bits):
        if bit == "1":
            x1 = (quiet + index) * module
            cv2.rectangle(image, (x1, 16), (x1 + module - 1, 172), (0, 0, 0), -1)
    cv2.putText(image, value, (quiet * module, 202), cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 0, 0), 2)
    return image


def synthetic_card(width: int = 680, height: int = 420) -> LabeledImage:
    canvas = np.full((height + 180, width + 180, 3), (228, 230, 234), np.uint8)
    x1, y1, x2, y2 = 90, 90, 90 + width, 90 + height
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (118, 72, 35), -1)
    cv2.rectangle(canvas, (x1 + 52, y1 + 102), (x1 + 166, y1 + 190), (190, 185, 92), -1)
    cv2.line(canvas, (x1 + 108, y1 + 102), (x1 + 108, y1 + 190), (70, 70, 40), 3)
    cv2.putText(canvas, "DEMO BANK", (x1 + 44, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (245, 245, 245), 3)
    cv2.putText(canvas, "4000 0000 0000 0002", (x1 + 45, y1 + 255), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (250, 250, 250), 3)
    cv2.putText(canvas, "TEST USER   12/30", (x1 + 45, y1 + 335), cv2.FONT_HERSHEY_SIMPLEX, .85, (245, 245, 245), 2)
    return LabeledImage("synthetic_payment_card", canvas, ((x1, y1, x2, y2),))


def synthetic_plate() -> LabeledImage:
    canvas = np.full((420, 900, 3), (95, 105, 115), np.uint8)
    box = (220, 155, 680, 285)
    cv2.rectangle(canvas, box[:2], box[2:], (245, 245, 245), -1)
    cv2.rectangle(canvas, box[:2], box[2:], (20, 20, 20), 5)
    cv2.putText(canvas, "ZZ 00 TEST", (250, 245), cv2.FONT_HERSHEY_DUPLEX, 1.75, (12, 12, 12), 4)
    return LabeledImage("synthetic_license_plate", canvas, (box,))


def synthetic_document() -> LabeledImage:
    canvas = np.full((760, 1100, 3), (232, 232, 232), np.uint8)
    box = (120, 90, 980, 670)
    cv2.rectangle(canvas, box[:2], box[2:], (247, 247, 242), -1)
    cv2.rectangle(canvas, box[:2], box[2:], (40, 80, 120), 5)
    cv2.putText(canvas, "SYNTHETIC ID - NOT VALID", (175, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (35, 35, 35), 3)
    cv2.rectangle(canvas, (170, 220), (380, 470), (140, 160, 180), -1)
    cv2.circle(canvas, (275, 300), 55, (90, 110, 135), -1)
    cv2.rectangle(canvas, (220, 355), (330, 450), (90, 110, 135), -1)
    lines = ("NAME: TEST USER", "ID: ABCDE1234F", "PHONE: 90000 00000", "PIN: 560001")
    for index, value in enumerate(lines):
        cv2.putText(canvas, value, (430, 265 + index * 80), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)
    qr = qr_image("id:test-only:ABCDE1234F", scale=3, border=12)
    qh, qw = qr.shape[:2]
    canvas[470:470 + qh, 740:740 + qw] = qr
    return LabeledImage("synthetic_identity_document", canvas, (box,))


def sensitive_text_image() -> tuple[np.ndarray, tuple[str, ...]]:
    lines = (
        "Phone: 90000 00000", "Email: test.user@example.com", "PAN: ABCDE1234F",
        "Aadhaar-like: 1111 2222 3333", "Address: 1 Demo Road", "Pincode: 560001",
        "Card-like: 4000 0000 0000 0002",
    )
    canvas = np.full((620, 1300, 3), 255, np.uint8)
    for index, value in enumerate(lines):
        cv2.putText(canvas, value, (45, 70 + index * 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2)
    return canvas, lines


def normal_text_image() -> np.ndarray:
    canvas = np.full((320, 1050, 3), 255, np.uint8)
    for index, value in enumerate(("Welcome to VVCE", "Computer Science Project", "Privacy Protection Demo")):
        cv2.putText(canvas, value, (40, 80 + index * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (25, 25, 25), 2)
    return canvas


def negative_shape_scene() -> np.ndarray:
    canvas = np.full((800, 1200, 3), 245, np.uint8)
    for x in (170, 480, 790, 1050):
        cv2.circle(canvas, (x, 230), 85, (180, 180, 180), -1)
        cv2.circle(canvas, (x - 28, 210), 10, (60, 60, 60), -1)
        cv2.circle(canvas, (x + 28, 210), 10, (60, 60, 60), -1)
    cv2.rectangle(canvas, (90, 430), (430, 650), (220, 220, 235), -1)
    cv2.putText(canvas, "PROJECT POSTER", (115, 555), cv2.FONT_HERSHEY_SIMPLEX, .9, (40, 40, 40), 2)
    cv2.rectangle(canvas, (510, 450), (750, 620), (210, 190, 170), -1)
    cv2.rectangle(canvas, (820, 440), (1110, 650), (190, 210, 220), -1)
    return canvas


def negative_card_scene(kind: str) -> np.ndarray:
    """Distinct rectangular non-card controls for card false-positive testing."""
    canvas = np.full((900, 1400, 3), (190, 175, 150), np.uint8)
    if kind == "phone":
        cv2.rectangle(canvas, (560, 245), (840, 700), (25, 25, 30), -1)
        cv2.rectangle(canvas, (580, 290), (820, 645), (70, 105, 135), -1)
    elif kind == "wallet":
        cv2.rectangle(canvas, (450, 335), (950, 650), (45, 70, 105), -1)
        cv2.circle(canvas, (860, 495), 18, (180, 180, 170), -1)
    elif kind == "id_card":
        cv2.rectangle(canvas, (430, 320), (970, 650), (235, 235, 225), -1)
        cv2.rectangle(canvas, (470, 375), (620, 555), (125, 145, 165), -1)
        cv2.putText(canvas, "DEMO MEMBER", (660, 430), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2)
    elif kind == "business_card":
        cv2.rectangle(canvas, (420, 350), (980, 650), (250, 250, 250), -1)
        cv2.putText(canvas, "VVCE PROJECT", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
        cv2.putText(canvas, "PRIVACY DEMO", (500, 530), cv2.FONT_HERSHEY_SIMPLEX, .8, (70, 70, 70), 2)
    elif kind == "notebook":
        cv2.rectangle(canvas, (390, 220), (1010, 720), (245, 242, 230), -1)
        for y in range(280, 690, 55):
            cv2.line(canvas, (450, y), (950, y), (150, 170, 190), 2)
    elif kind == "remote_control":
        cv2.rectangle(canvas, (590, 180), (810, 740), (55, 60, 65), -1)
        for y in range(260, 650, 80):
            cv2.circle(canvas, (700, y), 25, (160, 165, 170), -1)
    else:  # plain paper rectangle
        cv2.rectangle(canvas, (410, 260), (990, 700), (250, 250, 248), -1)
    return canvas


def difficult_privacy_scene() -> CompositeScene:
    """Build the Day 16 multi-risk scene without retaining any generated output."""
    people = load_fixture("synthetic_profile_occlusion_lowlight.png")
    canvas = cv2.resize(people, (1920, 1280), interpolation=cv2.INTER_AREA)

    card = synthetic_card()
    card_crop = card.pixels[90:510, 90:770]
    card_crop = cv2.resize(card_crop, (300, 185), interpolation=cv2.INTER_AREA)
    card_box = (810, 1035, 1110, 1220)
    canvas[card_box[1]:card_box[3], card_box[0]:card_box[2]] = card_crop

    document = synthetic_document()
    document_crop = document.pixels[90:670, 120:980]
    # The critical scene has one separately labeled QR; remove the document
    # factory's optional QR before compositing so the ground truth is unambiguous.
    document_crop[370:580, 600:860] = (247, 247, 242)
    document_crop = cv2.resize(document_crop, (390, 265), interpolation=cv2.INTER_AREA)
    document_box = (80, 980, 470, 1245)
    canvas[document_box[1]:document_box[3], document_box[0]:document_box[2]] = document_crop

    plate = synthetic_plate()
    plate_crop = plate.pixels[155:285, 220:680]
    plate_crop = cv2.resize(plate_crop, (260, 74), interpolation=cv2.INTER_AREA)
    plate_box = (1560, 1160, 1820, 1234)
    canvas[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]] = plate_crop

    qr = qr_image("upi://pay?pa=dummy@example.invalid&am=1", scale=4, border=16)
    qr = cv2.resize(qr, (150, 150), interpolation=cv2.INTER_NEAREST)
    qr_box = (1190, 1060, 1340, 1210)
    canvas[qr_box[1]:qr_box[3], qr_box[0]:qr_box[2]] = qr

    text_box = (1110, 955, 1810, 1025)
    cv2.rectangle(canvas, text_box[:2], text_box[2:], (248, 248, 248), -1)
    cv2.putText(
        canvas, "Phone: 90000 00000", (1130, 1005),
        cv2.FONT_HERSHEY_SIMPLEX, 1.35, (15, 15, 15), 3,
    )
    return CompositeScene(
        pixels=canvas,
        boxes={
            "faces": ((331, 262, 569, 669), (1300, 419, 1600, 819)),
            "license_plates": (plate_box,), "payment_cards": (card_box,),
            "identity_documents": (document_box,), "qr_codes": (qr_box,),
            # The document PIN line is a second real sensitive region, not an OCR
            # false positive, even though the dedicated document model is absent.
            "sensitive_text": (text_box, (218, 1157, 292, 1175)),
        },
    )
