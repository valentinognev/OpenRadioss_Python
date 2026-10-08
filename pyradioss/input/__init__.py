"""
pyradioss.input — reading Radioss decks.

Fortran origin: the Starter reader layer. In OpenRadioss the raw deck is
tokenized by the (C++) *hm_reader* library driven from
``starter/source/reader``; each keyword then has an ``hm_read_<keyword>.F``
routine that pulls typed fields out of the reader and fills the model
arrays. We split the job the same way:

    deck_reader.py       lexing: file -> list of KeywordBlock (also #include)
    keywords/            /NODE, /BRICK, /MAT/..., ...  -> Model
    engine_keywords.py   /RUN, /DT, /TFILE, /ANIM, ... -> EngineControls
"""

from .deck_reader import parse_fortran_float
from .deck_writer import write_engine_deck
from .engine_keywords import parse_engine_deck
