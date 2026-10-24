"""Run the pinned k23_analyze.main on Windows with the K23 WSL-share perception reader installed first (as k23_run does for plan / score)."""
import os, sys
M = r"D:\Personal\Master's Project\sim\final_test\matrix"
sys.path.insert(0, M)
import k23_entry as E
E.install_perception_reader()
import k23_analyze as KA
sys.exit(KA.main(sys.argv[1:]))
