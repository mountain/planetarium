# -*- coding: utf-8 -*-

import os


def link(src, tgt):
    #print("link:", src, tgt)
    hdsrc, tlsrc = os.path.split(src)
    hdtgt, tltgt = os.path.split(tgt)

    relpth = os.path.relpath(hdsrc, hdtgt)
    refpth = os.path.join(relpth, tlsrc)

    cwd = os.getcwd()
    os.chdir(hdtgt)
    try:
        os.symlink(refpth, tltgt)
    except Exception:
        pass
    os.chdir(cwd)
    assert os.path.isfile(tgt)
