"""Synthetic concept art for the tests — drawn in code, so nothing is fetched or committed."""
import io

import numpy as np
from PIL import Image, ImageDraw


def face():
    """A head-and-shoulders crop: deliberately *not* a full body, so the router
    must send it to the sculptor instead of the character rig."""
    im = Image.new('RGB', (512, 640), (210, 220, 235))
    d = ImageDraw.Draw(im)
    d.ellipse([120, 110, 400, 520], fill=(236, 190, 160))
    d.ellipse([100, 90, 420, 260], fill=(60, 42, 32))
    d.ellipse([190, 290, 240, 330], fill=(255, 255, 255))
    d.ellipse([285, 290, 335, 330], fill=(255, 255, 255))
    d.ellipse([205, 300, 228, 322], fill=(50, 40, 40))
    d.ellipse([300, 300, 322, 322], fill=(50, 40, 40))
    d.polygon([(255, 330), (270, 400), (240, 400)], fill=(220, 170, 145))
    d.arc([215, 400, 320, 470], 0, 180, fill=(160, 80, 80), width=8)
    return im


def logo():
    im = Image.new('RGBA', (640, 640), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([80, 80, 560, 560], fill=(230, 180, 60, 255))
    d.ellipse([160, 160, 480, 480], fill=(20, 20, 30, 255))
    d.rectangle([300, 180, 340, 460], fill=(20, 20, 30, 255))
    return im


def car():
    """A wide, flat side view — the case that exposes naive depth estimation."""
    im = Image.new('RGB', (900, 420), (225, 228, 236))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 180, 840, 320], 36, fill=(180, 30, 40))
    d.polygon([(200, 180), (300, 90), (620, 90), (700, 180)], fill=(150, 20, 30))
    d.ellipse([170, 260, 290, 380], fill=(30, 30, 34))
    d.ellipse([620, 260, 740, 380], fill=(30, 30, 34))
    d.ellipse([205, 295, 255, 345], fill=(190, 195, 205))
    d.ellipse([655, 295, 705, 345], fill=(190, 195, 205))
    d.rectangle([320, 110, 600, 170], fill=(150, 190, 220))
    return im


def knight():
    """A head-to-toe figure: the router should keep using the parametric rig."""
    im = Image.new('RGB', (520, 900), (238, 238, 242))
    d = ImageDraw.Draw(im)
    d.ellipse([190, 40, 330, 190], fill=(70, 52, 40))
    d.ellipse([215, 75, 305, 200], fill=(233, 186, 152))
    d.polygon([(150, 230), (370, 230), (340, 520), (180, 520)], fill=(90, 70, 160))
    d.polygon([(180, 520), (340, 520), (330, 800), (285, 800), (260, 640),
               (235, 800), (190, 800)], fill=(50, 55, 80))
    d.rectangle([200, 800, 250, 860], fill=(80, 60, 40))
    d.rectangle([280, 800, 330, 860], fill=(80, 60, 40))
    d.polygon([(150, 240), (100, 520), (135, 530), (185, 300)], fill=(233, 186, 152))
    d.polygon([(370, 240), (420, 520), (385, 530), (335, 300)], fill=(233, 186, 152))
    return im


def buf(im, fmt='PNG'):
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def open_edges(mesh):
    """Edges used by exactly one face — the honest measure of a leaky mesh."""
    e = np.sort(np.asarray(mesh.edges), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())
