#!/usr/bin/env python3
# validate_submission.py
"""
Standalone submission format validator.
Usage: python validate_submission.py --file submission.csv [--candidates candidates.jsonl]
"""
import argparse
from ranker.output import validate_submission_format

def main():
    parser = argparse.ArgumentParser(description="Redrob Submission Format Compliance Checker")
    parser.add_argument("--file", required=True, help="Path to your generated submission.csv")
    parser.add_argument("--candidates", default=None, help="Optional raw source database to match candidate IDs")
    
    args = parser.parse_args()
    report = validate_submission_format(args.file, args.candidates)
    
    if report["valid"]:
        print("✅ VALID — submission format passes all spec checks.")
    else:
        print("❌ INVALID:")
        for e in report["errors"]:
            print(f"    ERROR: {e}")
            
    if report["warnings"]:
        print("\n⚠️ WARNINGS:")
        for w in report["warnings"]:
            print(f"    WARN:  {w}")
            
    raise SystemExit(0 if report["valid"] else 1)

if __name__ == "__main__":
    main()