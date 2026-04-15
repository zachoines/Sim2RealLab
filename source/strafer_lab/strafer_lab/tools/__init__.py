"""Runtime helpers for the strafer_lab data pipeline.

This subpackage holds small utility modules shared between the batch
processing scripts and the Isaac Sim runtime. The Isaac Sim host agent
owns ``bbox_extractor.py`` and other runtime helpers; the DGX agent owns
the description/label/dataset utilities (:mod:`scene_labels`,
:mod:`spatial_description`, :mod:`dataset_export`).
"""
