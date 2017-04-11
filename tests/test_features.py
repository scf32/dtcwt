import sys, os
from pytest import raises

import numpy as np
from dtcwt.compat import dtwavexfm2, dtwaveifm2
from dtcwt.coeffs import biort, qshift
from dtcwt.features import slp2, slp2interleaved
import tests.datasets as datasets

TOLERANCE = 1e-12

def setup():
    global mandrill
    mandrill = datasets.mandrill()

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

def test_simple():
    slp2pyramid = slp2(mandrill)
    sampleLocs = slp2pyramid.init()
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)

def test_one_level():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=False, firstlevel=max_level, verbose=True)
    sampleLocs = slp2pyramid.init()
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    
def test_level_range():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=False, firstlevel=1, verbose=True)
    sampleLocs = slp2pyramid.init()
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    
def test_sampling_config():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='extended')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    
def test_hist_simple():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    hist = slp2pyramid.histgen(imgslp2)
    
def test_hist_level_range():
    slp2pyramid = slp2(mandrill, nlevels=2, full=False, firstlevel=2, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    hist = slp2pyramid.histgen(imgslp2, nbins=24, full=False)
    
def test_hist_nbins():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    hist = slp2pyramid.histgen(imgslp2, nbins=6, full=True)

def test_hist_best():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    hist = slp2pyramid.histgen(imgslp2, nbins=24, full=True, best=True)
    
def test_hist_vis():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    hist = slp2pyramid.histgen(imgslp2, nbins=24, full=True, best=True)
    slp2pyramid.histvis(hist[-1], 24)
    
def test_keypoints():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    slp2pyramid.keypoints(imgslp2, method='gale')
    hist = slp2pyramid.histgen(imgslp2, nbins=24, full=True, best=True)
    kps = slp2pyramid.keypoints(hist, method='forshaw', edge_suppression=1)
    slp2pyramid.draw_maps()
    slp2pyramid.draw_keypoints(kps, mandrill)
    
def test_nonsquare():
    max_level = 4
    slp2pyramid = slp2(mandrill[0:200,:], nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill[0:200,:], sampleLocs)
    slp2pyramid.keypoints(imgslp2, method='gale')
    hist = slp2pyramid.histgen(imgslp2, nbins=24, full=True, best=True)
    kps = slp2pyramid.keypoints(hist, method='forshaw', edge_suppression=1)
    slp2pyramid.draw_keypoints(kps, mandrill[0:200,:])
    
def test_interleaved():
    slp2interleaved(mandrill, nlevels=5, full=True, firstlevel=0)
    
def test_warp():
    max_level = 4
    slp2pyramid = slp2(mandrill, nlevels=max_level, full=True, firstlevel=0, verbose=True)
    sampleLocs = slp2pyramid.init(samplingConfig='normal')
    imgslp2, _ = slp2pyramid.transform(mandrill, sampleLocs)
    A = np.array([[1, -0.33, 50], [0.33, 1, 128], [0, 0, 1]])
    warpedSLP2 = slp2pyramid.global_warp(imgslp2, A, resample=True)
    