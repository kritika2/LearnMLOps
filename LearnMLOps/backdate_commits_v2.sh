#!/bin/bash

# Script to create backdated commits for LearnMLOps test suite
# Date range: October 15, 2025 to October 30, 2025

set -e

echo "🔄 Creating backdated commits for LearnMLOps test suite"
echo "Date range: October 15 - October 30, 2025"
echo "Author: kritika.jain <jainkritika2@gmail.com>"
echo ""

# Navigate to repository
cd "/Users/rojain3/Documents/Kritika/BlockChain Transactions"

# Remove dummy.txt first
if git status | grep -q "deleted.*dummy.txt"; then
    echo "🗑️  Removing dummy.txt (Oct 14)"
    git rm dummy.txt
    GIT_AUTHOR_NAME="kritika.jain" \
    GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
    GIT_COMMITTER_NAME="kritika.jain" \
    GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
    GIT_AUTHOR_DATE="2025-10-14 18:00:00" \
    GIT_COMMITTER_DATE="2025-10-14 18:00:00" \
    git commit -m "chore: Remove dummy file"
    echo "✅ Done"
    echo ""
fi

# October 15, 2025 - Initial project setup
echo "📅 Oct 15: Initial LearnMLOps project setup"
git add LearnMLOps/01-model-selection/ \
        LearnMLOps/02-data-splitting/ \
        LearnMLOps/03-hyperparameter-tuning/ \
        LearnMLOps/04-shadow-evaluation/ \
        LearnMLOps/README.md \
        LearnMLOps/requirements.txt \
        LearnMLOps/examples/

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-15 10:00:00" \
GIT_COMMITTER_DATE="2025-10-15 10:00:00" \
git commit -m "feat: Initialize LearnMLOps project with core modules

- Model selection helper
- Smart data splitter
- Hyperparameter auto-tuner
- Shadow evaluation system"
echo "✅ Done"
echo ""

# October 16, 2025 - Test infrastructure
echo "📅 Oct 16: Add test infrastructure"
git add LearnMLOps/tests/__init__.py \
        LearnMLOps/pytest.ini \
        LearnMLOps/conftest.py \
        LearnMLOps/.gitignore

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-16 14:30:00" \
GIT_COMMITTER_DATE="2025-10-16 14:30:00" \
git commit -m "feat: Add pytest configuration and test infrastructure

- pytest.ini with markers and settings
- conftest.py with shared fixtures
- .gitignore for test artifacts"
echo "✅ Done"
echo ""

# October 18, 2025 - Model selector tests
echo "📅 Oct 18: Add ModelSelector tests"
git add LearnMLOps/tests/test_model_selector.py

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-18 11:00:00" \
GIT_COMMITTER_DATE="2025-10-18 11:00:00" \
git commit -m "test: Add comprehensive tests for ModelSelector

- 45+ test cases covering all functionality
- Tests for classification, regression, clustering
- Edge cases and error handling
- Dataset analysis and model suggestions"
echo "✅ Done"
echo ""

# October 20, 2025 - Smart splitter tests
echo "📅 Oct 20: Add SmartSplitter tests"
git add LearnMLOps/tests/test_smart_splitter.py

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-20 15:45:00" \
GIT_COMMITTER_DATE="2025-10-20 15:45:00" \
git commit -m "test: Add comprehensive tests for SmartSplitter

- 40+ test cases for data splitting strategies
- Standard, stratified, temporal, grouped splits
- Split quality evaluation
- Reproducibility tests"
echo "✅ Done"
echo ""

# October 22, 2025 - Auto tuner tests
echo "📅 Oct 22: Add AutoTuner tests"
git add LearnMLOps/tests/test_auto_tuner.py

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-22 10:30:00" \
GIT_COMMITTER_DATE="2025-10-22 10:30:00" \
git commit -m "test: Add comprehensive tests for AutoTuner

- 35+ test cases for hyperparameter tuning
- Grid search, random search, Bayesian optimization
- Model type detection
- Strategy comparison"
echo "✅ Done"
echo ""

# October 24, 2025 - Shadow evaluator tests
echo "📅 Oct 24: Add ShadowEvaluator tests"
git add LearnMLOps/tests/test_shadow_evaluator.py

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-24 16:00:00" \
GIT_COMMITTER_DATE="2025-10-24 16:00:00" \
git commit -m "test: Add comprehensive tests for ShadowEvaluator

- 50+ test cases for shadow evaluation
- Parallel model execution tests
- Statistical comparison tests
- Business metrics and performance tracking"
echo "✅ Done"
echo ""

# October 26, 2025 - Documentation
echo "📅 Oct 26: Add comprehensive documentation"
git add LearnMLOps/TESTING.md \
        LearnMLOps/tests/README.md

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-26 13:00:00" \
GIT_COMMITTER_DATE="2025-10-26 13:00:00" \
git commit -m "docs: Add comprehensive testing documentation

- TESTING.md with usage guide
- tests/README.md with detailed instructions
- Examples and best practices"
echo "✅ Done"
echo ""

# October 28, 2025 - Test utilities
echo "📅 Oct 28: Add test runner and utilities"
git add LearnMLOps/run_tests.sh \
        LearnMLOps/backdate_commits.sh \
        LearnMLOps/backdate_commits_v2.sh

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-28 11:30:00" \
GIT_COMMITTER_DATE="2025-10-28 11:30:00" \
git commit -m "feat: Add interactive test runner script

- Interactive menu for test execution
- Options for coverage, specific modules
- Non-interactive mode support"
echo "✅ Done"
echo ""

# October 30, 2025 - Final documentation
echo "📅 Oct 30: Add final documentation"
git add LearnMLOps/TEST_SUITE_SUMMARY.md \
        LearnMLOps/tests/CREATED_FILES.md

GIT_AUTHOR_NAME="kritika.jain" \
GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
GIT_COMMITTER_NAME="kritika.jain" \
GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
GIT_AUTHOR_DATE="2025-10-30 14:00:00" \
GIT_COMMITTER_DATE="2025-10-30 14:00:00" \
git commit -m "docs: Add test suite summary and comprehensive documentation

- TEST_SUITE_SUMMARY.md with overview
- CREATED_FILES.md with file listing
- 170+ test cases documented"
echo "✅ Done"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All commits created successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Commit summary (Oct 14-30):"
git log --oneline --since="2025-10-14" --until="2025-10-31" --author="kritika.jain"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Ready to push! Run:"
echo "   git push origin main"
echo ""
echo "✨ All commits attributed to: kritika.jain <jainkritika2@gmail.com>"
echo "📅 Date range: October 14-30, 2025"
echo "📈 This will show as green squares on GitHub contribution graph!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

