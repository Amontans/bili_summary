# aicbili —— 零安装工作流
# 本项目除 Pillow 外无第三方依赖，系统 Python 一般已自带，所以不需要 pip install。

PY      ?= python3
PREFIX  ?= $(HOME)/.local
BIN     ?= $(PREFIX)/bin

.PHONY: help link unlink doctor test test-live run clean clean-all

help:
	@echo "make link        把 aicbili 命令链接到 $(BIN)（推荐，零安装）"
	@echo "make doctor      环境自检（ffmpeg / B站 API / 模型配置）"
	@echo "make test        跑单元测试（不需要联网、不需要 API Key）"
	@echo "make test-live   跑联网集成测试"
	@echo "make run U='BV1xx411c7mD'   直接分析一个视频"
	@echo "make clean       清理输出与缓存"
	@echo ""
	@echo "不装也能用：在当前目录直接跑 $(PY) -m aicbili <url>"

link:
	@mkdir -p $(BIN)
	@ln -sfn $(CURDIR)/bin/aicbili $(BIN)/aicbili
	@echo "✅ 已链接 $(BIN)/aicbili"
	@case ":$$PATH:" in *":$(BIN):"*) echo "   现在就能用：aicbili --version";; *) echo "   ⚠ $(BIN) 不在 PATH，把下面一行加到 ~/.bashrc："; echo "      export PATH=\"$(BIN):$$PATH\"";; esac

unlink:
	@rm -f $(BIN)/aicbili && echo "已移除 $(BIN)/aicbili"

doctor:
	@$(PY) -m aicbili doctor

test:
	@$(PY) -m unittest tests.test_core -v

test-live:
	@AICBILI_LIVE=1 $(PY) -m unittest tests.test_live -v

run:
	@test -n "$(U)" || { echo "用法：make run U='BV1xx411c7mD'"; exit 2; }
	@$(PY) -m aicbili $(U)

clean:
	@rm -rf output output_* .pytest_cache
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "已清理输出与缓存"

clean-all: clean
	@rm -rf .venv build dist *.egg-info
