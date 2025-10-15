"""
This script performs **visual reasoning–aware question fuzzification** through a two-stage pipeline, encapsulated in the `Fuzzifier` class. 
It aims to automatically rewrite multimodal questions into harder, answer-safe versions while maintaining solvability and structural integrity.

The pipeline consists of two main modules:

1. **Visual Condition Extraction**
    - Reads a metadata TSV file containing (question, answer, image_path).
    - For each entry, sends both textual and visual information to a multimodal LLM (e.g., `o3` or `gpt-4o`).
    - The model identifies key **visual evidence**, aligns it to text spans, and outputs a **structured JSON** containing:
        - localized bounding boxes (`bbox`) for evidence,
        - text-to-vision alignment mappings,
        - a detailed fuzzification plan (entity/property obfuscation),
        - and a rewritten “fuzzified question” that avoids answer leakage.
    - Aggregates all results and stores them in an intermediate JSON file.

2. **TSV Update (Question Replacement)**
    - Reads the intermediate JSON file.
    - Matches each record in the original TSV by image filename.
    - Replaces the original `question` field with the model-generated `fuzzified_question`.
    - Writes the updated dataset to a new TSV file while preserving all other metadata fields.

This process provides an **end-to-end pipeline** for constructing *harder, structurally consistent, and vision-grounded* versions of multimodal reasoning questions.

Example usage:
python LastBench_script/Agent_recognition.py \
  --input-tsv LastBench_data_without_txt-auto/metadata.tsv \
  --output-tsv LastBench_data_without_txt-auto_v4/metadata.tsv \
  --save-json LastBench_data_without_txt-auto_v4/model_outputs.json \
  --model o3
"""


from __future__ import annotations
import os
import io
import re
import json
import base64
import mimetypes
import argparse
import pprint
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from string import Template

import pandas as pd
from PIL import Image

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, W: int, H: int) -> "BBox":
        x = max(0, min(self.x, max(0, W - 1)))
        y = max(0, min(self.y, max(0, H - 1)))
        w = max(0, min(self.w, max(0, W - x)))
        h = max(0, min(self.h, max(0, H - y)))
        return BBox(x, y, w, h)


class Fuzzifier:    
    SYS_PROMPT = (
        "You are a strict multimodal grading and evidence extraction assistant. "
        "Given a question, a reference answer, and the corresponding image, you must:\n"
        "  (1) identify key visual evidence in the image supporting the correct answer, with tight pixel bboxes;\n"
        "  (2) align each evidence to concrete spans in the question text (symbols/labels/phrases);\n"
        "  (3) rewrite the question into a fuzzified version that:\n"
        "        • remains solvable from the same evidence;\n"
        "        • DOES NOT reveal the answer or equivalent hints;\n"
        "        • makes the problem harder by weakening direct references (e.g., AB, ∠XYZ, exact numbers) into relative or qualitative descriptors;\n"
        "        • **BUT must preserve the structural integrity and original format of the question**, "
        "          including options, numbering, or answer format (e.g., A/B/C/D, multiple choice list). "
        "          Do NOT delete or merge options; only modify descriptive content inside them when appropriate.\n"
        "  (4) perform consistency checks.\n"
        "Return JSON only.\n"
        "CRITICAL bbox policy: Each bbox should be of moderate size and clearly visible on the image. "
        "Avoid overly small boxes (hard to distinguish) and overly large boxes (covering the entire or most of the image). "
        "Bboxes must represent localized regions that are easy to observe and differentiate."
    )

    USER_INSTRUCTION_TMPL = Template(
        "Strictly output a single JSON object in the following schema:\n"
        "{\n"
        '  "conditions": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "description": "One-sentence description of a concrete, fine-grained visual detail",\n'
        '      "evidence": "Explain how this detail helps answer the question correctly",\n'
        '      "importance": "high|medium|low",\n'
        '      "confidence": 0.0,\n'
        '      "bbox": {"x": 0, "y": 0, "w": 0, "h": 0}\n'
        "    }\n"
        "  ],\n"
        '  "text_to_vision_alignment": [\n'
        "    {\n'      "
        '"text_span": "Original span in question (e.g., segment AB, point E)",\n'
        '      "evidence_id": 1,\n'
        '      "alignment_notes": "Why this evidence supports the span"\n'
        "    }\n"
        "  ],\n"
        '  "fuzzification_plan": {\n'
        '    "entity_obfuscation": [\n'
        "      {\n'        "
        '"original": "Original named entity (e.g., segment AB)",\n'
        '        "strategy": "rename|relative_ref|neighborhood|attribute_weaken",\n'
        '        "new_ref": "A more relative/qualitative reference (no direct label)"\n'
        "      }\n"
        "    ],\n"
        '    "property_obfuscation": [\n'
        "      {\n'        "
        '"original": "Original property (e.g., AB ⟂ CD or 90°)",\n'
        '        "strategy": "synonym|quant_to_qual|direction_blur",\n'
        '        "new_ref": "A weakened/qualitative property (e.g., close to a right angle)"\n'
        "      }\n"
        "    ]\n"
        "  },\n"
        '  "fuzzified_question": "Harder, answer-safe rephrasing of the question that preserves the original structure, numbering, and all options (e.g., A/B/C/D). Do NOT delete, merge, or omit any options; only modify descriptive wording to increase difficulty while keeping the question solvable.",\n'
        '  "leakage_check": {\n'
        '    "contains_answer_or_equivalents": false,\n'
        '    "notes": ""\n'
        "  },\n"
        '  "consistency_check": {\n'
        '    "supported_by_evidence_ids": [1],\n'
        '    "same_answer_expected": true,\n'
        '    "failure_modes_considered": ["naming conflict", "referential ambiguity"],\n'
        '    "confidence": 0.0\n'
        "  },\n"
        '  "notes": ""\n'
        "}\n\n"
        "Strict requirements:\n"
        f"1) Return at most $max_k conditions; sort by importance desc.\n"
        "2) Each bbox should be of moderate size and clearly visible on the image. Avoid overly small "
        "boxes (hard to distinguish) and overly large boxes (covering the entire or most of the image). "
        "Bboxes must represent localized regions that are easy to observe and differentiate.\n"
        "3) Fuzzification MUST make the problem harder, remain solvable from the same evidence, and avoid direct labels and exact numbers when possible.\n"
        "4) Absolutely DO NOT leak the answer or its equivalent paraphrases in the fuzzified question.\n"
        "5) The fuzzified question MUST retain the original structure and all answer options. Do not delete, merge, or renumber options. Only modify textual descriptions for difficulty adjustment.\n"
        "6) If multiple conditions, prefer spatially distinct details.\n\n"
        "question:\n$q\n\nanswer:\n$a\n"
    )

    def __init__(
        self,
        model: str = "o3",
        max_k: int = 3,
        temperature: float = 0.0,
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY"),
        base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL"),
    ):
        self.model = model
        self.max_k = max_k
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def run(
        self,
        input_tsv: str,
        output_tsv: str,
        save_json_path: Optional[str] = None,
    ):
        """
        Read TSV -> Call model -> Save JSON -> Update TSV.
        """
        print("Step 1: Extracting visual conditions from TSV and generating JSON...")
        json_records = self.extract_conditions_from_tsv(input_tsv, save_json_path)
        print(f"Generated {len(json_records)} records.")
        if save_json_path:
            print(f"Intermediate JSON saved to: {save_json_path}")

        print("\nStep 2: Updating TSV file using the generated JSON...")
        self.update_tsv_with_fuzzified_questions(
            json_records, input_tsv, output_tsv
        )
        print(f"Process completed. Output file: {output_tsv}")

    # ---------- Step 1: Extract conditions from TSV ----------
    def extract_conditions_from_tsv(
        self, tsv_path: str, save_json: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
        base_dir = os.path.dirname(os.path.abspath(tsv_path))
        image_dir = os.path.join(base_dir, "images")

        for col in ("question", "image_path", "answer"):
            if col not in df.columns:
                raise ValueError(f"TSV is missing required column: {col}")

        all_records: List[Dict[str, Any]] = []

        for i, row in df.iterrows():
            print(f"Processing: {i + 1}/{len(df)}...")
            q = row["question"]
            a = row["answer"]
            img_path = row["image_path"]

            if not img_path:
                all_records.append({"question": q, "answer": a, "image_path": "", "skipped": True})
                continue

            if not os.path.isabs(img_path):
                img_path = os.path.normpath(os.path.join(image_dir, img_path))

            try:
                _, raw_text, payload = self._extract_one(q, img_path, a)
                record = {
                    "question": q, "answer": a, "image_path": img_path,
                    "raw_text": raw_text, "payload": payload,
                }
            except (FileNotFoundError, ValueError, APIError, RateLimitError, APITimeoutError) as e:
                record = {"question": q, "answer": a, "image_path": img_path, "error": str(e)}
            except Exception as e:
                record = {"question": q, "answer": a, "image_path": img_path, "error": f"unexpected: {repr(e)}"}

            all_records.append(record)

        if save_json:
            os.makedirs(os.path.dirname(save_json), exist_ok=True)
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)

        return all_records

    # ---------- Step 2: Update TSV ----------
    def update_tsv_with_fuzzified_questions(
        self,
        records: List[Dict[str, Any]],
        input_tsv_path: str,
        output_tsv_path: str,
    ):
        df = pd.read_csv(input_tsv_path, sep="\t", dtype=str, keep_default_na=False)
        required_cols = {"question", "image_path"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"metadata.tsv is missing required columns: {required_cols - set(df.columns)}")

        df["_image_basename"] = df["image_path"].apply(self._basename_norm)
        index_map: Dict[str, List[int]] = {
            bname: group.index.tolist()
            for bname, group in df.groupby("_image_basename")
        }

        updated, skipped_error, skipped_no_fq, matched_rows = 0, 0, 0, 0

        for rec in records:
            if rec.get("error"):
                skipped_error += 1
                continue

            img_path = rec.get("image_path", "")
            bname = self._basename_norm(img_path)
            if not bname:
                skipped_no_fq += 1
                continue

            fq = self._try_extract_fuzzified_question(rec)
            if not fq:
                skipped_no_fq += 1
                continue

            row_idxs = index_map.get(bname, [])
            if not row_idxs:
                continue

            for ridx in row_idxs:
                df.at[ridx, "question"] = fq
                matched_rows += 1
            updated += 1

        df.drop(columns=["_image_basename"], inplace=True)

        if os.path.abspath(output_tsv_path) == os.path.abspath(input_tsv_path):
            bak = input_tsv_path + ".bak"
            if not os.path.exists(bak):
                os.rename(input_tsv_path, bak)
                print(f"[Backup] Original TSV has been backed up to: {bak}")

        os.makedirs(os.path.dirname(os.path.abspath(output_tsv_path)), exist_ok=True)
        df.to_csv(output_tsv_path, sep="\t", index=False, encoding="utf-8")

        print(f"[Done] Written to: {output_tsv_path}")
        print(f"  Applied JSON records: {updated}")
        print(f"  Matched and updated TSV rows: {matched_rows}")
        print(f"  Skipped (with error): {skipped_error}")
        print(f"  Skipped (no fuzzified_question or image_path): {skipped_no_fq}")

    def _extract_one(self, question: str, image_path: str, answer: str) -> Tuple[List[BBox], str, Dict[str, Any]]:
        b64, W, H, mime = self._load_image_b64(image_path)
        raw_text = self._call_model(b64, mime, question, answer)
        payload = self._extract_json(raw_text)
        payload = self._sanitize_payload(payload, W, H, self.max_k)
        boxes = self._payload_to_bboxes(payload, W, H)
        if not boxes:
            boxes = [BBox(0, 0, min(32, W), min(32, H))]
        return boxes, raw_text, payload

    @staticmethod
    def _load_image_b64(path: str) -> Tuple[str, int, int, str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        mime, _ = mimetypes.guess_type(path) or ("image/png", None)
        with Image.open(path) as im:
            W, H = im.size
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            fmt = mime.split("/")[-1].upper()
            try:
                im.save(buf, format=fmt)
            except Exception:
                buf, mime = io.BytesIO(), "image/png"
                im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, W, H, mime

    def _call_model(self, b64: str, mime: str, question: str, answer: str) -> str:
        user_prompt = self.USER_INSTRUCTION_TMPL.substitute(max_k=self.max_k, q=question, a=answer)
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=4096,
                response_format={"type": "json_object"}, messages=messages
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=4096, messages=messages
            )
        return resp.choices[0].message.content

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
            raise ValueError("Model did not return valid JSON.")

    @staticmethod
    def _sanitize_payload(payload: Dict[str, Any], W: int, H: int, max_k: int = 3) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "conditions": [],
            "text_to_vision_alignment": [],
            "fuzzification_plan": {"entity_obfuscation": [], "property_obfuscation": []},
            "fuzzified_question": "",
            "leakage_check": {"contains_answer_or_equivalents": False, "notes": ""},
            "consistency_check": {
                "supported_by_evidence_ids": [],
                "same_answer_expected": True,
                "failure_modes_considered": ["naming conflict", "referential ambiguity"],
                "confidence": 0.0
            },
            "notes": ""
        }

        if not isinstance(payload, dict):
            return out

        # notes
        out["notes"] = payload.get("notes", "") if isinstance(payload.get("notes", ""), str) else ""

        # conditions
        conds = payload.get("conditions", [])
        if not isinstance(conds, list):
            conds = []

        sanitized_conds: List[Dict[str, Any]] = []
        for i, c in enumerate(conds[:max_k], start=1):
            if not isinstance(c, dict):
                continue
            desc = str(c.get("description", "")).strip()
            ev = str(c.get("evidence", "")).strip()
            imp = c.get("importance", "high")
            if imp not in ("high", "medium", "low"):
                imp = "high"
            try:
                conf = float(c.get("confidence", 0.0))
            except Exception:
                conf = 0.0

            bb = c.get("bbox", {}) or {}
            try:
                x = int(bb.get("x", 0))
                y = int(bb.get("y", 0))
                w = int(bb.get("w", 0))
                h = int(bb.get("h", 0))
            except Exception:
                x = y = w = h = 0

            x = max(0, min(x, max(0, W - 1)))
            y = max(0, min(y, max(0, H - 1)))
            w = max(0, min(w, max(0, W - x)))
            h = max(0, min(h, max(0, H - y)))

            sanitized_conds.append({
                "id": int(c.get("id", i)),
                "description": desc,
                "evidence": ev,
                "importance": imp,
                "confidence": conf,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
            })

        if not sanitized_conds:
            sanitized_conds = [{
                "id": 1,
                "description": "no-condition-returned",
                "evidence": "",
                "importance": "high",
                "confidence": 0.0,
                "bbox": {"x": 0, "y": 0, "w": min(32, W), "h": min(32, H)}
            }]
        out["conditions"] = sanitized_conds

        # alignment
        align = payload.get("text_to_vision_alignment", [])
        if isinstance(align, list):
            sanitized_align = []
            for a in align:
                if not isinstance(a, dict):
                    continue
                text_span = str(a.get("text_span", "")).strip()
                try:
                    evidence_id = int(a.get("evidence_id", 1))
                except Exception:
                    evidence_id = 1
                notes = str(a.get("alignment_notes", "")).strip()
                sanitized_align.append({
                    "text_span": text_span,
                    "evidence_id": evidence_id,
                    "alignment_notes": notes
                })
            out["text_to_vision_alignment"] = sanitized_align

        # fuzzification_plan
        fp = payload.get("fuzzification_plan", {}) or {}
        ent = fp.get("entity_obfuscation", []) if isinstance(fp, dict) else []
        prop = fp.get("property_obfuscation", []) if isinstance(fp, dict) else []
        ent_s, prop_s = [], []
        if isinstance(ent, list):
            for e in ent:
                if not isinstance(e, dict):
                    continue
                ent_s.append({
                    "original": str(e.get("original", "")),
                    "strategy": str(e.get("strategy", "")),
                    "new_ref": str(e.get("new_ref", "")),
                })
        if isinstance(prop, list):
            for p in prop:
                if not isinstance(p, dict):
                    continue
                prop_s.append({
                    "original": str(p.get("original", "")),
                    "strategy": str(p.get("strategy", "")),
                    "new_ref": str(p.get("new_ref", "")),
                })
        out["fuzzification_plan"] = {"entity_obfuscation": ent_s, "property_obfuscation": prop_s}

        # fuzzified_question
        fq = payload.get("fuzzified_question", "")
        out["fuzzified_question"] = str(fq) if isinstance(fq, str) else ""

        # leakage_check
        leak = payload.get("leakage_check", {}) or {}
        cae = bool(leak.get("contains_answer_or_equivalents", False)) if isinstance(leak, dict) else False
        notes = str(leak.get("notes", "")) if isinstance(leak, dict) else ""
        out["leakage_check"] = {"contains_answer_or_equivalents": cae, "notes": notes}

        # consistency_check
        cc = payload.get("consistency_check", {}) or {}
        sup = cc.get("supported_by_evidence_ids", [])
        if not isinstance(sup, list):
            sup = []
        sae = bool(cc.get("same_answer_expected", True))
        fmc = cc.get("failure_modes_considered", ["naming conflict", "referential ambiguity"])
        if not isinstance(fmc, list):
            fmc = ["naming conflict", "referential ambiguity"]
        try:
            conf = float(cc.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        out["consistency_check"] = {
            "supported_by_evidence_ids": sup,
            "same_answer_expected": sae,
            "failure_modes_considered": fmc,
            "confidence": conf
        }
        return out

    @staticmethod
    def _payload_to_bboxes(payload: Dict[str, Any], W: int, H: int) -> List[BBox]:
        boxes: List[BBox] = []
        for c in payload.get("conditions", []):
            if isinstance(c, dict) and "bbox" in c:
                bb = c["bbox"]
                boxes.append(BBox(bb.get("x",0), bb.get("y",0), bb.get("w",0), bb.get("h",0)).clamp(W, H))
        return boxes

    @staticmethod
    def _try_extract_fuzzified_question(rec: Dict[str, Any]) -> Optional[str]:
        if isinstance(rec.get("payload"), dict):
            fq = rec["payload"].get("fuzzified_question")
            if isinstance(fq, str) and fq.strip():
                return fq.strip()
        if isinstance(rec.get("raw_text"), str):
            m = re.search(r'["\']fuzzified_question["\']\s*:\s*["\'](.*?)["\']', rec["raw_text"], re.S)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _basename_norm(p: str) -> str:
        return os.path.basename(str(p)).strip().lower()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent-based Visual Condition Extraction and Update")
    
    # model params
    parser.add_argument("--model", type=str, default="o3", help="model to use (e.g., o3, gpt-4o, etc.)")
    parser.add_argument("--max-k", type=int, default=3, help="The maximum number of conditions to extract")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model sampling")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI API Base URL")

    # path params
    parser.add_argument("--input-tsv", required=True, help="The input tsv file path")
    parser.add_argument("--output-tsv", required=True, help="The output tsv file path")
    parser.add_argument("--save-json", type=str, help="The path to save intermediate JSON outputs (optional)")

    args = parser.parse_args()

    fuzzifier = Fuzzifier(
        model=args.model,
        max_k=args.max_k,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    fuzzifier.run(
        input_tsv=args.input_tsv,
        output_tsv=args.output_tsv,
        save_json_path=args.save_json,
    )
