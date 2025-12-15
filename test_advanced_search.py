"""
Test Advanced Search Endpoint

Tests the /search/advanced endpoint with specific test cases to verify:
1. "what is covered in document" → Routes to Summary
2. "git staging area explanation" → Routes to Section (excludes summary)
3. "git working directory and staging area" → Routes to Section(s)

This validates the enhanced query analyzer fixes.
"""

import requests
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ExpectedRouting(Enum):
    """Expected routing behavior."""
    SUMMARY = "summary"
    SECTION = "section"
    MIXED = "mixed"


@dataclass
class TestCase:
    """Test case definition."""
    query: str
    expected_routing: ExpectedRouting
    expected_scope: str  # "document_level", "section_level", "mixed"
    expected_strategy: str  # "summary_first", "section_only", "hybrid"
    should_exclude_summary: bool
    description: str


class AdvancedSearchTester:
    """Test the advanced search endpoint."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.advanced_search_url = f"{base_url}/search/advanced"
        self.results: List[Dict[str, Any]] = []

    def test_case(self, test: TestCase) -> Dict[str, Any]:
        """
        Run a single test case.

        Args:
            test: TestCase to execute

        Returns:
            Test result dictionary
        """
        print(f"\n{'='*80}")
        print(f"Test: {test.description}")
        print(f"Query: '{test.query}'")
        print(f"Expected: {test.expected_routing.value} routing")
        print(f"{'='*80}")

        try:
            # Call the advanced search endpoint
            response = requests.post(
                self.advanced_search_url,
                json={
                    "query": test.query,
                    "min_results": 1,
                    "max_results": 10
                },
                timeout=30
            )

            if response.status_code != 200:
                return self._create_error_result(test, f"HTTP {response.status_code}: {response.text}")

            data = response.json()

            # Extract analysis
            analysis = data.get('analysis', {})
            scope = analysis.get('scope')
            strategy = analysis.get('search_strategy')
            should_exclude = analysis.get('should_exclude_summary', False)
            summary_only = analysis.get('summary_only', False)
            confidence = analysis.get('confidence', 0.0)

            # Extract results info
            total_results = data.get('total_results', 0)
            summary_excluded_count = data.get('summary_excluded_count', 0)
            fallback_used = data.get('fallback_used', False)
            results = data.get('results', [])

            # Print analysis
            print(f"\n📊 Analysis:")
            print(f"   Scope: {scope}")
            print(f"   Strategy: {strategy}")
            print(f"   Should exclude summary: {should_exclude}")
            print(f"   Summary only: {summary_only}")
            print(f"   Confidence: {confidence:.2f}")

            # Print results info
            print(f"\n📈 Results:")
            print(f"   Total results: {total_results}")
            print(f"   Summaries excluded: {summary_excluded_count}")
            print(f"   Fallback used: {fallback_used}")

            # Validate expectations
            validations = self._validate_test_case(
                test, scope, strategy, should_exclude,
                total_results, summary_excluded_count, results
            )

            # Print validation results
            print(f"\n✅ Validation:")
            all_passed = True
            for validation in validations:
                status = "✓" if validation['passed'] else "✗"
                print(f"   {status} {validation['check']}: {validation['message']}")
                if not validation['passed']:
                    all_passed = False

            # Print first few results
            if results:
                print(f"\n📄 Top Results:")
                for i, result in enumerate(results[:3], 1):
                    section = result.get('payload', {}).get('section_title', 'Unknown')
                    score = result.get('score', 0.0)
                    text_preview = result.get('payload', {}).get('text', '')[:100]
                    print(f"   [{i}] Score: {score:.3f} | Section: {section}")
                    print(f"       {text_preview}...")

            # Overall status
            status = "PASSED ✅" if all_passed else "FAILED ❌"
            print(f"\n{'='*80}")
            print(f"Test Status: {status}")
            print(f"{'='*80}")

            result = {
                'test': test.description,
                'query': test.query,
                'passed': all_passed,
                'analysis': analysis,
                'total_results': total_results,
                'summary_excluded_count': summary_excluded_count,
                'fallback_used': fallback_used,
                'validations': validations
            }

            self.results.append(result)
            return result

        except requests.exceptions.ConnectionError:
            return self._create_error_result(test, "Connection failed - is the server running?")
        except requests.exceptions.Timeout:
            return self._create_error_result(test, "Request timeout")
        except Exception as e:
            return self._create_error_result(test, f"Unexpected error: {str(e)}")

    def _validate_test_case(
        self,
        test: TestCase,
        scope: str,
        strategy: str,
        should_exclude: bool,
        total_results: int,
        summary_excluded_count: int,
        results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Validate test case expectations."""
        validations = []

        # Check scope
        scope_passed = scope == test.expected_scope
        validations.append({
            'check': 'Scope Detection',
            'passed': scope_passed,
            'message': f"Expected '{test.expected_scope}', got '{scope}'"
        })

        # Check strategy
        strategy_passed = strategy == test.expected_strategy
        validations.append({
            'check': 'Search Strategy',
            'passed': strategy_passed,
            'message': f"Expected '{test.expected_strategy}', got '{strategy}'"
        })

        # Check summary exclusion
        exclusion_passed = should_exclude == test.should_exclude_summary
        validations.append({
            'check': 'Summary Exclusion',
            'passed': exclusion_passed,
            'message': f"Expected {test.should_exclude_summary}, got {should_exclude}"
        })

        # Check results exist
        has_results = total_results > 0
        validations.append({
            'check': 'Has Results',
            'passed': has_results,
            'message': f"Found {total_results} result(s)"
        })

        # For section routing, verify no summaries in results
        if test.should_exclude_summary and results:
            summaries_found = [
                r for r in results
                if r.get('payload', {}).get('section_title') == '[DOCUMENT SUMMARY]'
            ]
            no_summaries = len(summaries_found) == 0
            validations.append({
                'check': 'No Summaries in Results',
                'passed': no_summaries,
                'message': f"Found {len(summaries_found)} summary chunk(s) in results"
            })

        # For summary routing, verify summaries are prioritized
        if test.expected_routing == ExpectedRouting.SUMMARY and results:
            top_result = results[0]
            is_summary = top_result.get('payload', {}).get('section_title') == '[DOCUMENT SUMMARY]'
            validations.append({
                'check': 'Summary Prioritized',
                'passed': is_summary or total_results == 0,
                'message': f"Top result is {'summary' if is_summary else 'not summary'}"
            })

        return validations

    def _create_error_result(self, test: TestCase, error: str) -> Dict[str, Any]:
        """Create error result."""
        print(f"\n❌ ERROR: {error}")
        result = {
            'test': test.description,
            'query': test.query,
            'passed': False,
            'error': error
        }
        self.results.append(result)
        return result

    def run_all_tests(self, test_cases: List[TestCase]):
        """Run all test cases and generate summary."""
        print("\n" + "="*80)
        print("ADVANCED SEARCH ENDPOINT - TEST SUITE")
        print("="*80)

        for test in test_cases:
            self.test_case(test)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed = sum(1 for r in self.results if r.get('passed', False))
        failed = len(self.results) - passed

        print(f"\nTotal Tests: {len(self.results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")

        if failed > 0:
            print("\n⚠️  Failed Tests:")
            for result in self.results:
                if not result.get('passed', False):
                    print(f"   - {result['test']}")
                    if 'error' in result:
                        print(f"     Error: {result['error']}")
                    elif 'validations' in result:
                        for v in result['validations']:
                            if not v['passed']:
                                print(f"     ✗ {v['check']}: {v['message']}")

        print("\n" + "="*80)


def main():
    """Run the test suite."""

    # Define test cases based on user's feedback
    test_cases = [
        TestCase(
            query="what is covered in document",
            expected_routing=ExpectedRouting.SUMMARY,
            expected_scope="document_level",
            expected_strategy="summary_first",
            should_exclude_summary=False,
            description="Test Case 1: Document overview query → Summary"
        ),
        TestCase(
            query="git staging area explanation",
            expected_routing=ExpectedRouting.SECTION,
            expected_scope="section_level",
            expected_strategy="section_only",
            should_exclude_summary=True,
            description="Test Case 2: Specific topic explanation → Section (exclude summary)"
        ),
        TestCase(
            query="git working directory and staging area",
            expected_routing=ExpectedRouting.SECTION,
            expected_scope="section_level",
            expected_strategy="section_only",
            should_exclude_summary=True,
            description="Test Case 3: Multi-concept query → Sections"
        ),
        # Additional edge cases
        TestCase(
            query="overview of git cheat sheet",
            expected_routing=ExpectedRouting.SUMMARY,
            expected_scope="document_level",
            expected_strategy="summary_first",
            should_exclude_summary=False,
            description="Test Case 4: Document overview → Summary"
        ),
        TestCase(
            query="how to commit changes",
            expected_routing=ExpectedRouting.SECTION,
            expected_scope="section_level",
            expected_strategy="section_only",
            should_exclude_summary=True,
            description="Test Case 5: How-to query → Section"
        ),
        TestCase(
            query="difference between merge and rebase",
            expected_routing=ExpectedRouting.SECTION,
            expected_scope="section_level",
            expected_strategy="section_only",
            should_exclude_summary=True,
            description="Test Case 6: Comparison query → Section"
        ),
    ]

    # Run tests
    tester = AdvancedSearchTester()
    tester.run_all_tests(test_cases)

    # Save results to file
    with open('test_advanced_search_results.json', 'w') as f:
        json.dump(tester.results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: test_advanced_search_results.json")


if __name__ == "__main__":
    main()
