"""Standalone preview of just the DetPO mAP page (no heavy model imports)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detpo_map_browser import render_detpo_map_page
render_detpo_map_page()
