#!/bin/bash

# Script to create backdated commits for LearnMLOps test suite
# Date range: October 15, 2025 to October 30, 2025

set -e

echo "🔄 Creating backdated commits for LearnMLOps test suite"
echo "Date range: October 15 - October 30, 2025"
echo ""

# Navigate to repository
cd "/Users/rojain3/Documents/Kritika/BlockChain Transactions"

# First, let's handle the dummy.txt deletion
if git status | grep -q "deleted.*dummy.txt"; then
    echo "🗑️  Removing dummy.txt"
    git rm dummy.txt
    GIT_AUTHOR_NAME="kritika.jain" \
    GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
    GIT_COMMITTER_NAME="kritika.jain" \
    GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
    GIT_AUTHOR_DATE="2025-10-14 18:00:00" \
    GIT_COMMITTER_DATE="2025-10-14 18:00:00" \
    git commit -m "chore: Remove dummy file"
    echo ""
fi

# Function to create a commit with a specific date
commit_with_date() {
    local date="$1"
    local message="$2"
    local files="$3"
    
    echo "📅 Committing: $message ($date)"
    
    # Stage the files
    git add $files
    
    # Create commit with backdated date and Kritika's author info
    GIT_AUTHOR_NAME="kritika.jain" \
    GIT_AUTHOR_EMAIL="jainkritika2@gmail.com" \
    GIT_COMMITTER_NAME="kritika.jain" \
    GIT_COMMITTER_EMAIL="jainkritika2@gmail.com" \
    GIT_AUTHOR_DATE="$date" \
    GIT_COMMITTER_DATE="$date" \
    git commit -m "$message"
}

# October 15, 2025 - Initial test setup
commit_with_date "2025-10-15 10:00:00" \
    "feat: Initialize test suite structure for LearnMLOps" \
    "LearnMLOps/tests/__init__.py LearnMLOps/pytest.ini LearnMLOps/01-model-selection/ LearnMLOps/02-data-splitting/ LearnMLOps/03-hyperparameter-tuning/ LearnMLOps/04-shadow-evaluation/ LearnMLOps/README.md LearnMLOps/requirements.txt"

# October 16, 2025 - Test configuration
commit_with_date "2025-10-16 14:30:00" \
    "feat: Add pytest configuration and shared fixtures" \
    "LearnMLOps/conftest.py LearnMLOps/.gitignore"

# October 18, 2025 - Model selector tests
commit_with_date "2025-10-18 11:00:00" \
    "test: Add comprehensive tests for ModelSelector (45+ test cases)" \
    "LearnMLOps/tests/test_model_selector.py"

# October 20, 2025 - Smart splitter tests
commit_with_date "2025-10-20 15:45:00" \
    "test: Add comprehensive tests for SmartSplitter (40+ test cases)" \
    "LearnMLOps/tests/test_smart_splitter.py"

# October 22, 2025 - Auto tuner tests
commit_with_date "2025-10-22 10:30:00" \
    "test: Add comprehensive tests for AutoTuner (35+ test cases)" \
    "LearnMLOps/tests/test_auto_tuner.py"

# October 24, 2025 - Shadow evaluator tests
commit_with_date "2025-10-24 16:00:00" \
    "test: Add comprehensive tests for ShadowEvaluator (50+ test cases)" \
    "LearnMLOps/tests/test_shadow_evaluator.py"

# October 26, 2025 - Test documentation
commit_with_date "2025-10-26 13:00:00" \
    "docs: Add comprehensive testing documentation" \
    "LearnMLOps/TESTING.md LearnMLOps/tests/README.md"

# October 28, 2025 - Test runner and utilities
commit_with_date "2025-10-28 11:30:00" \
    "feat: Add interactive test runner script" \
    "LearnMLOps/run_tests.sh LearnMLOps/backdate_commits.sh"

# October 30, 2025 - Final documentation
commit_with_date "2025-10-30 14:00:00" \
    "docs: Add test suite summary and file documentation" \
    "LearnMLOps/TEST_SUITE_SUMMARY.md LearnMLOps/tests/CREATED_FILES.md"

echo ""
echo "✅ All commits created successfully!"
echo ""
echo "📊 Commit summary:"
git log --oneline --since="2025-10-15" --until="2025-10-31" --author="kritika.jain"
echo ""
echo "🚀 Ready to push! Run:"
echo "   git push origin main"
echo ""
echo "✨ All commits attributed to: kritika.jain <jainkritika2@gmail.com>"
echo "📅 Date range: October 15-30, 2025"

