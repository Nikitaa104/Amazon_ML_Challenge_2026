import csv
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Set, Tuple


def check_candidate_subset_invariant(
    candidate_pairs_path: Path, matching_results_path: Path
) -> Tuple[bool, List[str]]:
    """Sanity check to confirm every matched ID appears in the candidate_pairs list for that S1 entity.
    
    Args:
        candidate_pairs_path: Path to candidate_pairs.tsv file.
        matching_results_path: Path to matching_results.tsv file.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not candidate_pairs_path.exists():
        errors.append(f"File not found: {candidate_pairs_path.resolve()}")
        return False, errors

    if not matching_results_path.exists():
        errors.append(f"File not found: {matching_results_path.resolve()}")
        return False, errors

    # 1. Parse candidate_pairs.tsv
    cand_map: Dict[str, Set[str]] = {}
    cand_s1_order: List[str] = []

    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2 or header[0] != "source1_entity_id":
            errors.append(f"Invalid header in candidate_pairs.tsv: {header}")
            return False, errors

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            cand_str = row[1].strip() if len(row) > 1 else ""

            if s1_id in cand_map:
                errors.append(f"Duplicate source1_entity_id '{s1_id}' on line {row_idx} of candidate_pairs.tsv")

            c_ids = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str else []
            
            # Check duplicates in candidate list
            if len(c_ids) != len(set(c_ids)):
                errors.append(f"Duplicate candidate IDs found in list for S1 '{s1_id}' in candidate_pairs.tsv")
                
            # Check self match
            if s1_id in c_ids:
                errors.append(f"Source 1 ID '{s1_id}' appears in its own candidate list in candidate_pairs.tsv")

            cand_map[s1_id] = set(c_ids)
            cand_s1_order.append(s1_id)

    # 2. Parse matching_results.tsv
    match_map: Dict[str, Set[str]] = {}
    match_s1_order: List[str] = []

    with open(matching_results_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2 or header[0] != "source1_entity_id":
            errors.append(f"Invalid header in matching_results.tsv: {header}")
            return False, errors

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            match_str = row[1].strip() if len(row) > 1 else ""

            if s1_id in match_map:
                errors.append(f"Duplicate source1_entity_id '{s1_id}' on line {row_idx} of matching_results.tsv")

            m_ids = [m.strip() for m in match_str.split(",") if m.strip()] if match_str else []

            # Check duplicates in match list
            if len(m_ids) != len(set(m_ids)):
                errors.append(f"Duplicate matched IDs found in list for S1 '{s1_id}' in matching_results.tsv")

            # Check self match
            if s1_id in m_ids:
                errors.append(f"Source 1 ID '{s1_id}' appears in its own matched list in matching_results.tsv")

            match_map[s1_id] = set(m_ids)
            match_s1_order.append(s1_id)

    # 3. Check entity list alignment
    if cand_s1_order != match_s1_order:
        errors.append("Mismatch in Source 1 entity order or set between candidate_pairs.tsv and matching_results.tsv")

    # 4. Check candidate subset invariant (Every matched ID MUST appear in candidate list)
    for s1_id, m_set in match_map.items():
        c_set = cand_map.get(s1_id, set())
        missing_from_candidates = m_set - c_set
        if missing_from_candidates:
            errors.append(
                f"Validation Failure for S1 '{s1_id}': Matched ID(s) {missing_from_candidates} "
                f"never appeared in candidate_pairs.tsv!"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def run_external_validator(
    validator_script: Path, test_dir: Path, matching_file: Path, candidate_file: Path
) -> Tuple[bool, str]:
    """Execute external utils/validate_submission.py validator script via subprocess.
    
    Args:
        validator_script: Path to validate_submission.py script.
        test_dir: Path to dataset/test directory.
        matching_file: Path to output/matching_results.tsv.
        candidate_file: Path to output/candidate_pairs.tsv.
        
    Returns:
        Tuple of (success_bool, stdout_stderr_output).
    """
    cmd = [
        sys.executable,
        str(validator_script),
        "--matching",
        str(matching_file),
        "--candidate",
        str(candidate_file),
        "--test-dir",
        str(test_dir),
    ]


    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout + "\n" + proc.stderr
        return (proc.returncode == 0), output.strip()
    except Exception as e:
        return False, f"Failed to execute validator script: {str(e)}"


def validate(
    output_dir: str = "output",
    dataset_dir: str = "dataset",
    utils_dir: str = "utils",
) -> bool:
    """Run local sanity checks and external utils/validate_submission.py validation script.
    
    Args:
        output_dir: Path to output directory containing TSV files.
        dataset_dir: Path to dataset directory containing test/ directory.
        utils_dir: Path to utils directory containing validate_submission.py.
        
    Returns:
        True if all validations pass, False otherwise.
    """
    print("=" * 60)
    print("SUBMISSION VALIDATION & SANITY CHECK")
    print("=" * 60)

    out_path = Path(output_dir)
    test_path = Path(dataset_dir) / "test"
    if not test_path.exists():
        for alt in [Path("../../dataset/test"), Path("../dataset/test")]:
            if alt.exists():
                test_path = alt
                break

    cand_file = out_path / "candidate_pairs.tsv"
    match_file = out_path / "matching_results.tsv"

    # 1. Local invariant sanity check
    print("\n1. Running Local Candidate Subset Invariant Check...")
    is_valid, errors = check_candidate_subset_invariant(cand_file, match_file)

    if is_valid:
        print("  [SUCCESS] All matched IDs strictly appear in candidate lists.")
        print("  [SUCCESS] Entity counts, formatting, and non-self-matching invariants satisfied.")
    else:
        print("  [FAILURE] Validation errors detected:")
        for err in errors:
            print(f"    - {err}")
        return False

    # 2. External validator check
    print("\n2. Searching for utils/validate_submission.py...")
    val_script_candidates = [
        Path(utils_dir) / "validate_submission.py",
        Path("../../utils/validate_submission.py"),
        Path("../utils/validate_submission.py"),
        Path("utils/validate_submission.py"),
    ]

    found_script = None
    for s in val_script_candidates:
        if s.exists():
            found_script = s
            break

    if found_script:
        print(f"  Executing validator: {found_script.resolve()}")
        success, output = run_external_validator(
            found_script, test_path, match_file, cand_file
        )
        if success:
            print("  [SUCCESS] utils/validate_submission.py passed cleanly!")
            print(f"  Validator Output:\n{output}")
        else:
            print("  [FAILURE] utils/validate_submission.py reported validation failures:")
            print(output)
            return False
    else:
        print("  [NOTE] utils/validate_submission.py not found at relative path.")
        print("  Local invariant checks passed cleanly.")

    print("\n" + "=" * 60)
    print("ALL VALIDATION CHECKS PASSED READY FOR SUBMISSION")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
