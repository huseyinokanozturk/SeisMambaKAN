# SeisMambaKAN — convenience wrappers around `python run.py`.
# These are most useful inside Colab / Linux. Windows users should call
# `python run.py <cmd>` directly.

PYTHON ?= python
MODE   ?= all
EPOCHS ?=
EXP    ?=
SPLIT  ?= val

.PHONY: help setup data train eval infer tb status push pull-exp git-pull clean-data clean-exp

help:
	@echo "SeisMambaKAN — common tasks"
	@echo
	@echo "  make setup MODE=all       # Colab bootstrap (mount, clone, deps, data)"
	@echo "  make data MODE=all        # Sync processed data from Drive"
	@echo "  make train EPOCHS=30      # Train (override epochs)"
	@echo "  make eval EXP=7 SPLIT=test"
	@echo "  make infer EXP=7"
	@echo "  make tb [EXP=7]"
	@echo "  make status"
	@echo "  make push EXP=7           # Mirror exp_007 + results to Drive"
	@echo "  make pull-exp EXP=7       # Pull exp_007 from Drive"
	@echo "  make git-pull"

setup:
	$(PYTHON) run.py setup --data-mode $(MODE)

data:
	$(PYTHON) run.py data --mode $(MODE)

train:
	$(PYTHON) run.py train $(if $(EPOCHS),--epochs $(EPOCHS),)

eval:
	$(PYTHON) run.py eval $(if $(EXP),--exp $(EXP),) --split $(SPLIT)

infer:
	$(PYTHON) run.py infer $(if $(EXP),--exp $(EXP),) --split $(SPLIT)

tb:
	$(PYTHON) run.py tb $(if $(EXP),--exp $(EXP),)

status:
	$(PYTHON) run.py status

push:
	@test -n "$(EXP)" || (echo "Usage: make push EXP=<id>"; exit 2)
	$(PYTHON) run.py push --exp $(EXP)

pull-exp:
	@test -n "$(EXP)" || (echo "Usage: make pull-exp EXP=<id>"; exit 2)
	$(PYTHON) run.py pull-exp --exp $(EXP)

git-pull:
	git pull --rebase

clean-data:
	rm -rf data/processed

clean-exp:
	rm -rf experiments results
