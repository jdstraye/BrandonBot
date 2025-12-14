Documentation:
- [ ] Results: a report with nightly validation runs. It should have any pytest failures but also the latest judges' scores. It should have a plot on the front page with the average judge score on the y-axis and the date/time on the x-axis. In order to create it, you probably need to create a nightly cron job to launch the full validation suite.
 - [x] SRS: Create this as if you were creating an SRS for someone else to implement what we currently have and are implementing. (scaffolded under `docs/SRS.md`)
 - [x] Validation Plan: Create this as if you were creating an SRS for someone else to implement what we currently have and are implementing. (scaffolded under `docs/VALIDATION_PLAN.md`)
 - [x] Developer Guide: How the code and execution is organized. Help for a person new to the project to jump in and make a positive impact. The current challenges and opportunities. (scaffolded under `docs/DEVELOPER_GUIDE.md`)
 - [x] Architecture Document containing: (scaffolded under `docs/ARCHITECTURE.md`)
  - [ ] high-level overview, including validated UML diagrams
  - [ ] function-by-function crawl
  - [ ] current challenges
  - [ ] future enhancement opportunities
  - [ ] Needed maintenance/updates (e.g., any RAG databases need refreshed with the latest information?)
 - [x] Review: a line-by-line explanation of the code - what it does, how, and why (who consumes its output, who produces its input) as well as needed or opportunistic enhancements (e.g., models needing fine-tuning or files that need updated). It's a huge file. (scaffolded under `docs/CODE_REVIEW.md`)
 - [x] Update `scripts/install_refresh_cron.sh` to support `--install-all` and `--print-nightly` to install/print nightly validation cron as part of a single-step install.
- [ ] Add open-mistral-nemo-2407 as a second slot for Mistral.
- [ ] Add mcp tools expected of a political chatbot, and recommended system prompts.
- [ ] replit vs. aws hosting expenses
- [ ] adaptive scoring/whitelist updater
- [x] Add an 8-hour cache to SearXNG.
- [x] TinyLlama-1.1B is a very compact model (1.1 billion parameters) available in quantized versions, which makes it a strong candidate for running on a desktop with an i7 processor. While the search results don't explicitly mention TinyLlama's MCP capabilities, its small size and open-source nature make it a candidate worth considering for agent-like tasks
- [x] Another model to consider is Glaive-function-calling-v1, a 2.7 billion parameter model built on Replit's Code Model. It's designed to handle function-calling tasks and can generate JSON responses, which aligns well with MCP requirements. This model has shown capabilities comparable to larger models like GPT-3.5 and GPT-4 in certain tasks
- [x] Function Calling Mistral 7B is also noteworthy, as it enhances the HuggingFace Instruct model with function-calling abilities, producing structured JSON outputs. However, its larger size (7 billion parameters) might be a consideration for a desktop setup
- [ ] Use global variable class instead of environment variables, e.g., os.environ.get("JUDGE_MODEL", "llama3.2:3b")
- [ ] Deprecation warnings:
  - [x]   /home/cana/cana/BrandonBot.git/backend/.venv_brandonbot/lib/python3.12/site-packages/pypdf/_crypt_providers/_cryptography.py:32: CryptographyDeprecationWarning: ARC4 has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.ARC4 and will be removed from cryptography.hazmat.primitives.ciphers.algorithms in 48.0.0. from cryptography.hazmat.primitives.ciphers.algorithms import AES, ARC4
  - [x]   tests/test_ov_intent.py:12 /home/cana/cana/BrandonBot.git/backend/tests/test_ov_intent.py:12: DeprecationWarning: There is no current event loop `loop = asyncio.get_event_loop()`
  - [ ]     /home/cana/cana/BrandonBot.git/backend/.venv_brandonbot/lib/python3.12/site-packages/PyPDF2/__init__.py:21: DeprecationWarning: PyPDF2 is deprecated. Please move to the pypdf library instead.
    warnings.warn(
        -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- [ ] Pylance errors
- [x] Upgrade `pypdf` / `cryptography` to address ARC4 deprecation warnings (see test output: CryptographyDeprecationWarning from pypdf internals)
 - [x] Update README.md and developer_guide.md in great detail about project structure, architecture, and validation testing. (added quick-start cron install instructions)


Great. It's time for another git checkpoint. Please suggest a commit message.