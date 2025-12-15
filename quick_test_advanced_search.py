"""
Quick Test - Advanced Search Endpoint

A simplified test script for quick validation of the three main test cases.
"""

import requests
import json


def quick_test(base_url="http://localhost:8007"):
    """Run quick tests on the three main test cases."""

    test_queries = [
        {
            "query": "what is covered in document",
            "expected": "Summary (document-level)",
            "should_exclude_summary": False
        },
        {
            "query": "git staging area explanation",
            "expected": "Section (exclude summary)",
            "should_exclude_summary": True
        },
        {
            "query": "git working directory and staging area",
            "expected": "Sections (multi-concept)",
            "should_exclude_summary": True
        }
    ]

    print("="*80)
    print("QUICK TEST - Advanced Search Endpoint")
    print("="*80)

    results = []

    for i, test in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: '{test['query']}'")
        print(f"Expected: {test['expected']}")
        print("-"*80)

        try:
            response = requests.post(
                f"{base_url}/search/advanced",
                json={"query": test['query']},
                timeout=10
            )

            if response.status_code != 200:
                print(f"❌ Error: HTTP {response.status_code}")
                results.append({"query": test['query'], "status": "error"})
                continue

            data = response.json()
            analysis = data.get('analysis', {})

            # Print key info
            print(f"Scope: {analysis.get('scope')}")
            print(f"Strategy: {analysis.get('search_strategy')}")
            print(f"Exclude Summary: {analysis.get('should_exclude_summary')}")
            print(f"Total Results: {data.get('total_results')}")
            print(f"Summaries Excluded: {data.get('summary_excluded_count')}")
            print(f"Fallback Used: {data.get('fallback_used')}")

            # Check if expectations met
            exclude_match = analysis.get('should_exclude_summary') == test['should_exclude_summary']
            has_results = data.get('total_results', 0) > 0

            if exclude_match and has_results:
                print("✅ PASSED")
                status = "passed"
            elif not has_results:
                print("⚠️  WARNING: No results found")
                status = "no_results"
            elif not exclude_match:
                print("❌ FAILED: Summary exclusion mismatch")
                status = "failed"
            else:
                print("❌ FAILED")
                status = "failed"

            results.append({
                "query": test['query'],
                "status": status,
                "analysis": analysis,
                "total_results": data.get('total_results')
            })

        except requests.exceptions.ConnectionError:
            print("❌ Error: Connection failed - is the server running?")
            results.append({"query": test['query'], "status": "connection_error"})
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({"query": test['query'], "status": "error"})

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r['status'] == 'passed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    warnings = sum(1 for r in results if r['status'] == 'no_results')
    errors = sum(1 for r in results if r['status'] in ['error', 'connection_error'])

    print(f"Passed: {passed}/3 ✅")
    print(f"Failed: {failed}/3 ❌")
    print(f"Warnings: {warnings}/3 ⚠️")
    print(f"Errors: {errors}/3")

    if passed == 3:
        print("\n🎉 All tests passed!")
    elif failed > 0:
        print("\n⚠️  Some tests failed - review output above")

    return results


if __name__ == "__main__":
    results = quick_test()

    # Save results
    with open('quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: quick_test_results.json")
