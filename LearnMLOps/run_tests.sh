#!/bin/bash

# LearnMLOps Test Runner Script
# This script provides convenient ways to run the test suite

echo "🧪 LearnMLOps Test Suite Runner"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if pytest is installed
check_pytest() {
    if ! command -v pytest &> /dev/null; then
        echo -e "${RED}❌ pytest is not installed${NC}"
        echo "Please install it with: pip install pytest"
        exit 1
    fi
}

# Function to display menu
show_menu() {
    echo ""
    echo "Select test option:"
    echo "1. Run all tests"
    echo "2. Run tests with coverage"
    echo "3. Run specific module tests"
    echo "4. Run tests in verbose mode"
    echo "5. Run quick tests (skip slow ones)"
    echo "6. Exit"
    echo ""
}

# Run all tests
run_all_tests() {
    echo -e "${YELLOW}Running all tests...${NC}"
    pytest
}

# Run tests with coverage
run_with_coverage() {
    echo -e "${YELLOW}Running tests with coverage...${NC}"
    if command -v pytest-cov &> /dev/null; then
        pytest --cov=. --cov-report=html --cov-report=term
        echo -e "${GREEN}✅ Coverage report generated in htmlcov/index.html${NC}"
    else
        echo -e "${RED}pytest-cov not installed. Installing...${NC}"
        pip install pytest-cov
        pytest --cov=. --cov-report=html --cov-report=term
    fi
}

# Run specific module tests
run_specific_module() {
    echo ""
    echo "Select module to test:"
    echo "1. Model Selection"
    echo "2. Smart Splitter"
    echo "3. Auto Tuner"
    echo "4. Shadow Evaluator"
    echo "5. Back to main menu"
    echo ""
    read -p "Enter choice [1-5]: " module_choice
    
    case $module_choice in
        1)
            echo -e "${YELLOW}Running Model Selection tests...${NC}"
            pytest tests/test_model_selector.py -v
            ;;
        2)
            echo -e "${YELLOW}Running Smart Splitter tests...${NC}"
            pytest tests/test_smart_splitter.py -v
            ;;
        3)
            echo -e "${YELLOW}Running Auto Tuner tests...${NC}"
            pytest tests/test_auto_tuner.py -v
            ;;
        4)
            echo -e "${YELLOW}Running Shadow Evaluator tests...${NC}"
            pytest tests/test_shadow_evaluator.py -v
            ;;
        5)
            return
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
}

# Run tests in verbose mode
run_verbose() {
    echo -e "${YELLOW}Running tests in verbose mode...${NC}"
    pytest -v -s
}

# Run quick tests
run_quick() {
    echo -e "${YELLOW}Running quick tests (skipping slow tests)...${NC}"
    pytest -m "not slow"
}

# Main script
check_pytest

if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        show_menu
        read -p "Enter choice [1-6]: " choice
        
        case $choice in
            1)
                run_all_tests
                ;;
            2)
                run_with_coverage
                ;;
            3)
                run_specific_module
                ;;
            4)
                run_verbose
                ;;
            5)
                run_quick
                ;;
            6)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Non-interactive mode with arguments
    case $1 in
        all)
            run_all_tests
            ;;
        coverage)
            run_with_coverage
            ;;
        verbose)
            run_verbose
            ;;
        quick)
            run_quick
            ;;
        model_selector)
            pytest tests/test_model_selector.py -v
            ;;
        smart_splitter)
            pytest tests/test_smart_splitter.py -v
            ;;
        auto_tuner)
            pytest tests/test_auto_tuner.py -v
            ;;
        shadow_evaluator)
            pytest tests/test_shadow_evaluator.py -v
            ;;
        *)
            echo "Usage: $0 [all|coverage|verbose|quick|model_selector|smart_splitter|auto_tuner|shadow_evaluator]"
            exit 1
            ;;
    esac
fi

