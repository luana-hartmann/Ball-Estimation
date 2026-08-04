# Makefile for the ball detection pipeline.
# Run from the repo root (where this file lives).
#
# Structure this expects:
#   src/                current pipeline code
#   data/<name>/         video + markup per video, e.g. data/test_2/test_2.mp4
#   outputs/trajectories/  generated CSVs
#   outputs/eval/           generated per-frame eval CSVs
#   outputs/diagnosis/      generated diagnostic crops
#
# ---------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------
# Run the full pipeline for a video:
#   make all NAME=test2
#
# Run just one step:
#   make trajectory NAME=test2
#   make eval        NAME=test2
#   make diagnose    NAME=test2
#
# Override a threshold for one run without editing this file:
#   make all NAME=test2 MIN_REL_V=8
#
# Remove generated files for one video (force a clean rerun):
#   make clean NAME=test2
# ---------------------------------------------------------------------

# ---- shared detector thresholds: THE single source of truth ----
# Calibrated on test_2 only (src/calibrate.py) -- verified to generalize
# reasonably to test_5 and test_6 without retuning (see report).
MIN_AREA       ?= 4
MAX_AREA       ?= 650
MAX_ASPECT     ?= 3.2
MIN_REL_V      ?= 5.0
MAX_REL_S      ?= -4.0
MAX_PRED_DIST  ?= 60
EVAL_THRESHOLD ?= 10

# ---- per-run identifiers ----
NAME   ?= test2
VIDEO  ?= data/$(NAME)/$(NAME).mp4
MARKUP ?= data/$(NAME)/ball_markup_$(NAME).json

TRAJECTORY    = outputs/trajectories/trajectory_$(NAME).csv
EVAL_PERFRAME = outputs/eval/eval_per_frame_$(NAME).csv
DIAG_DIR      = outputs/diagnosis/fn_diagnosis_$(NAME)
WRONGPOS_DIR  = outputs/diagnosis/wrong_pos_diagnosis_$(NAME)

.PHONY: all trajectory eval diagnose diagnose_wrong_pos clean help dirs

help:
	@echo "Targets: trajectory, eval, diagnose, diagnose_wrong_pos, all, clean"
	@echo "Usage: make all NAME=test2   (expects data/test2/test2.mp4 + data/test2/ball_markup_test2.json)"

dirs:
	@mkdir -p outputs/trajectories outputs/eval outputs/diagnosis

trajectory: $(TRAJECTORY)

$(TRAJECTORY): | dirs
	python src/detector.py \
		--video $(VIDEO) \
		--out $(TRAJECTORY) \
		--min_area $(MIN_AREA) \
		--max_area $(MAX_AREA) \
		--max_aspect_ratio $(MAX_ASPECT) \
		--min_rel_v $(MIN_REL_V) \
		--max_rel_s $(MAX_REL_S) \
		--max_pred_dist $(MAX_PRED_DIST)

eval: $(EVAL_PERFRAME)
	python src/summarize_eval.py --eval_csv $(EVAL_PERFRAME)

$(EVAL_PERFRAME): $(TRAJECTORY) | dirs
	python src/evaluate.py \
		--pred $(TRAJECTORY) \
		--markup $(MARKUP) \
		--threshold $(EVAL_THRESHOLD) \
		--per_frame_out $(EVAL_PERFRAME)

diagnose: $(EVAL_PERFRAME)
	python src/diagnose_fn.py \
		--video $(VIDEO) \
		--eval_csv $(EVAL_PERFRAME) \
		--min_area $(MIN_AREA) \
		--max_area $(MAX_AREA) \
		--max_aspect_ratio $(MAX_ASPECT) \
		--min_rel_v $(MIN_REL_V) \
		--max_rel_s $(MAX_REL_S) \
		--out_dir $(DIAG_DIR)

diagnose_wrong_pos: $(EVAL_PERFRAME)
	python src/diagnose_wrong_pos.py \
		--video $(VIDEO) \
		--eval_csv $(EVAL_PERFRAME) \
		--out_dir $(WRONGPOS_DIR)

all: eval
	@echo "Done. Trajectory: $(TRAJECTORY)  |  Per-frame eval: $(EVAL_PERFRAME)"

clean:
	rm -f outputs/trajectories/trajectory_$(NAME).csv outputs/eval/eval_per_frame_$(NAME).csv
	rm -rf $(DIAG_DIR) $(WRONGPOS_DIR)