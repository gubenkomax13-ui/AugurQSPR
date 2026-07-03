#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.spectra_core import *  # noqa: F401,F403


RESULT_COLUMNS = [
    "source_line_number",
    "compound_id",
    "name",
    "cas",
    "input_smiles",
    "canonical_smiles",
    "inchikey",
    "structure_status",
    "spectrum_type",
    "spectrum_status",
    "selected_source",
    "candidate_count",
    "spectrum_id",
    "raw_file",
    "processed_file",
    "message",
    "candidate_url",
    "_from_real_search",
    "_task_seconds",
]


def append_log(paths, message):
    try:
        os.makedirs(paths["job_dir"], exist_ok=True)
        with open(paths["log"], "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")
    except Exception:
        pass


def normalize_search_result(compound, spectrum_type_norm, result):
    normalized = {
        "source_line_number": compound.get("source_line_number", compound.get("row_index", "")),
        "compound_id": compound.get("compound_id", ""),
        "name": compound.get("name", ""),
        "cas": compound.get("cas", ""),
        "input_smiles": compound.get("input_smiles", ""),
        "canonical_smiles": compound.get("canonical_smiles", ""),
        "inchikey": compound.get("inchikey", ""),
        "structure_status": compound.get("structure_status", ""),
        "spectrum_type": spectrum_type_norm,
        "spectrum_status": "search_error",
        "selected_source": "",
        "candidate_count": 0,
        "spectrum_id": "",
        "raw_file": "",
        "processed_file": "",
        "message": "",
        "candidate_url": "",
        "_from_real_search": True,
    }

    if not isinstance(result, dict):
        normalized["message"] = "search returned non-dict result"
        return normalized

    status_value = (
        result.get("spectrum_status", "")
        or result.get("status", "")
        or result.get("final_status", "")
    )

    normalized["spectrum_status"] = status_value or "search_error"
    normalized["message"] = (
        result.get("message", "")
        or result.get("status_message", "")
        or result.get("error", "")
    )
    normalized["selected_source"] = (
        result.get("selected_source", "")
        or result.get("source_database", "")
        or result.get("source", "")
    )
    normalized["candidate_count"] = (
        result.get("candidate_count", 0)
        or result.get("n_candidates", 0)
        or result.get("candidates_count", 0)
    )
    normalized["spectrum_id"] = result.get("spectrum_id", "") or result.get("id", "")
    normalized["raw_file"] = (
        result.get("raw_file", "")
        or result.get("raw_path", "")
        or result.get("raw_jdx_path", "")
        or result.get("downloaded_file", "")
        or result.get("file_path", "")
    )
    normalized["processed_file"] = (
        result.get("processed_file", "")
        or result.get("processed_path", "")
        or result.get("processed_csv", "")
    )
    normalized["candidate_url"] = result.get("candidate_url", "")

    for record_key in ["record", "spectrum_record", "index_record", "saved_record"]:
        record = result.get(record_key, None)

        if not isinstance(record, dict):
            continue

        normalized["selected_source"] = (
            normalized["selected_source"]
            or record.get("selected_source", "")
            or record.get("source_database", "")
            or record.get("source", "")
        )
        normalized["spectrum_id"] = (
            normalized["spectrum_id"]
            or record.get("spectrum_id", "")
            or record.get("id", "")
        )
        normalized["raw_file"] = (
            normalized["raw_file"]
            or record.get("raw_file", "")
            or record.get("raw_path", "")
            or record.get("raw_jdx_path", "")
            or record.get("downloaded_file", "")
            or record.get("file_path", "")
        )
        normalized["processed_file"] = (
            normalized["processed_file"]
            or record.get("processed_file", "")
            or record.get("processed_path", "")
            or record.get("processed_csv", "")
        )

    return normalized


def should_cache_result(result_row):
    result_status = str(result_row.get("spectrum_status", "")).strip()
    result_message = str(result_row.get("message", "")).strip()

    if result_status == "stopped_by_user":
        return False

    if result_status in [
        "found_downloaded",
        "not_found_in_all_sources",
        "candidate_link_found",
        "already_in_bank",
        "parse_error",
        "download_error",
        "no_numeric_spectrum",
    ]:
        return True

    if result_status == "search_error":
        return "is not defined" not in result_message and "NameError" not in result_message

    return False


def write_results(paths, skipped_results, search_results):
    rows = []
    rows.extend(skipped_results or [])
    rows.extend(search_results or [])

    df = pd.DataFrame(rows)

    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    os.makedirs(paths["job_dir"], exist_ok=True)
    df.to_csv(paths["results"], index=False, encoding="utf-8-sig")
    return df


def run_job(job_id):
    paths = spectra_worker_paths(job_id)

    with open(paths["config"], "r", encoding="utf-8") as f:
        config = json.load(f)

    selected_sources = config.get("selected_sources", [])
    delay_seconds = float(config.get("delay_seconds", 0.5) or 0.5)
    max_workers = max(1, int(config.get("max_workers", 4) or 4))
    source_timeout = float(config.get("source_timeout_seconds", 10) or 10)
    cache_flush_every = max(1, int(config.get("cache_flush_every", 25) or 25))

    spectra_set_http_timeout(source_timeout)
    spectra_clear_stop()

    tasks_df = pd.read_csv(paths["tasks"], dtype=str, low_memory=False).fillna("")

    if os.path.exists(paths["skipped"]):
        skipped_df = pd.read_csv(paths["skipped"], dtype=str, low_memory=False).fillna("")
        skipped_results = skipped_df.to_dict(orient="records")
    else:
        skipped_results = []

    tasks = []
    for _, row in tasks_df.iterrows():
        compound = row.to_dict()
        spectrum_type = spectra_normalize_spectrum_type(compound.pop("_task_spectrum_type", ""))
        tasks.append((compound, spectrum_type))

    total_tasks = len(tasks)
    done_tasks = 0
    search_results = []
    cache_buffer = []
    started_at = time.perf_counter()
    status_counts = {}

    def update_status(state="running", last_result=None):
        elapsed = time.perf_counter() - started_at
        tasks_per_minute = done_tasks / (elapsed / 60.0) if elapsed > 0 and done_tasks else 0.0
        payload = {
            "state": state,
            "done_tasks": done_tasks,
            "total_tasks": total_tasks,
            "skipped_tasks": len(skipped_results),
            "result_rows": len(skipped_results) + len(search_results),
            "elapsed_seconds": elapsed,
            "tasks_per_minute": tasks_per_minute,
            "avg_seconds_per_task": elapsed / done_tasks if done_tasks else 0.0,
            "max_workers": max_workers,
            "source_timeout_seconds": source_timeout,
            "selected_sources": selected_sources,
            "status_counts": status_counts,
            "results_file": paths["results"],
            "log_file": paths["log"],
        }
        if last_result:
            payload["last_result"] = last_result
        spectra_write_worker_status(job_id, payload)

    def flush_cache(force=False):
        if not cache_buffer:
            return

        if not force and len(cache_buffer) < cache_flush_every:
            return

        rows = list(cache_buffer)
        cache_buffer.clear()
        spectra_add_many_to_search_cache(rows, selected_sources)

    def search_task(compound, spectrum_type):
        if spectra_is_stop_requested():
            return {
                "spectrum_status": "stopped_by_user",
                "message": "Search stopped before task started.",
            }

        result = spectra_search_one_compound(
            compound=compound,
            spectrum_type=spectrum_type,
            selected_sources=selected_sources,
            delay_seconds=delay_seconds,
        )
        return normalize_search_result(compound, spectrum_type, result)

    append_log(paths, f"started job {job_id}; tasks={total_tasks}; workers={max_workers}")
    write_results(paths, skipped_results, search_results)
    update_status("running")

    with ThreadPoolExecutor(max_workers=min(max_workers, max(total_tasks, 1))) as executor:
        task_iter = iter(tasks)
        future_to_task = {}

        def submit_next():
            if spectra_is_stop_requested():
                return False
            try:
                compound_next, spectrum_type_next = next(task_iter)
            except StopIteration:
                return False
            future = executor.submit(search_task, compound_next, spectrum_type_next)
            future_to_task[future] = (compound_next, spectrum_type_next, time.perf_counter())
            return True

        for _ in range(min(max_workers, total_tasks)):
            submit_next()

        while future_to_task:
            done_futures, _ = wait(
                list(future_to_task.keys()),
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )

            if not done_futures:
                update_status("running")
                continue

            for future in done_futures:
                compound, spectrum_type, task_started = future_to_task.pop(future)
                task_seconds = time.perf_counter() - task_started

                try:
                    result_row = future.result()
                except Exception as exc:
                    result_row = normalize_search_result(
                        compound,
                        spectrum_type,
                        {"spectrum_status": "search_error", "message": str(exc)},
                    )

                result_row["_task_seconds"] = round(task_seconds, 3)
                search_results.append(result_row)
                done_tasks += 1

                status = str(result_row.get("spectrum_status", "")).strip() or "unknown"
                status_counts[status] = int(status_counts.get(status, 0)) + 1

                if should_cache_result(result_row):
                    cache_buffer.append(result_row)
                    flush_cache(False)

                if done_tasks == 1 or done_tasks % 10 == 0:
                    write_results(paths, skipped_results, search_results)

                update_status("running", result_row)

                if spectra_is_stop_requested():
                    continue

                submit_next()

    flush_cache(True)
    write_results(paths, skipped_results, search_results)

    final_state = "stopped_by_user" if spectra_is_stop_requested() else "completed"
    update_status(final_state)
    append_log(paths, f"finished job {job_id}; state={final_state}; done={done_tasks}/{total_tasks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    try:
        run_job(args.job_id)
    except Exception as exc:
        try:
            spectra_write_worker_status(
                args.job_id,
                {
                    "state": "failed",
                    "error": str(exc),
                },
            )
        finally:
            raise


if __name__ == "__main__":
    main()
