.PHONY: install install-dev scenario sumo-gui sumo-gui-low sumo-gui-medium sumo-gui-high \
        train-dqn train-ppo train-mappo eval test lint format demo clean

UV ?= uv
SCENARIO_DIR = sumo_scenarios/highway_3lane

# --- Setup ----------------------------------------------------------------

install:
	$(UV) sync --extra sumo --extra rl --extra multiagent

install-dev:
	$(UV) sync --extra sumo --extra rl --extra multiagent --extra dev

# --- SUMO scenario --------------------------------------------------------

scenario:
	cd $(SCENARIO_DIR) && bash build.sh

sumo-gui: sumo-gui-medium

sumo-gui-low:
	sumo-gui -c $(SCENARIO_DIR)/highway.sumocfg --route-files $(SCENARIO_DIR)/routes_low.rou.xml

sumo-gui-medium:
	sumo-gui -c $(SCENARIO_DIR)/highway.sumocfg --route-files $(SCENARIO_DIR)/routes_medium.rou.xml

sumo-gui-high:
	sumo-gui -c $(SCENARIO_DIR)/highway.sumocfg --route-files $(SCENARIO_DIR)/routes_high.rou.xml

# --- Training -------------------------------------------------------------

train-dqn:
	$(UV) run python -m scripts.train_dqn

train-ppo:
	$(UV) run python -m scripts.train_ppo

train-mappo:
	$(UV) run python -m scripts.train_mappo

# --- Evaluation -----------------------------------------------------------

eval:
	$(UV) run python -m scripts.eval_all

# --- Quality gates --------------------------------------------------------

test:
	$(UV) run pytest -m "not slow"

test-all:
	$(UV) run pytest

lint:
	$(UV) run ruff check laneiq tests scripts

format:
	$(UV) run ruff format laneiq tests scripts
	$(UV) run ruff check --fix laneiq tests scripts

# --- Demo -----------------------------------------------------------------

demo:
	@echo "Demo: start backend in one terminal, frontend in another:"
	@echo "  (terminal 1) cd backend && $(UV) run uvicorn app.main:app --reload"
	@echo "  (terminal 2) cd frontend && pnpm dev"

clean:
	rm -rf runs/* checkpoints/*.pt checkpoints/*.zip
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
