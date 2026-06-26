

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from skin_disease_data import SKIN_DISEASE_DATABASE, get_disease_info

# ── Load medicine database ────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH  = os.path.join(_BASE_DIR, "medicine_database.json")


def _load_medicine_db() -> dict:
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[druggpt_engine] WARNING: could not load medicine_database.json: {exc}")
        return {"medicines": {}, "disease_medications": {}, "no_medication_notes": {}}


_DB = _load_medicine_db()

MEDICINE_DATABASE:   Dict[str, dict]        = _DB.get("medicines", {})
DISEASE_MEDICATIONS: Dict[str, List[str]]   = _DB.get("disease_medications", {})
NO_MEDICATION_NOTES: Dict[str, str]         = _DB.get("no_medication_notes", {})

SAFETY_DISCLAIMER = (
    "⚕️ DrugGPT provides general educational information only — not a medical "
    "diagnosis or prescription. Dosing and suitability vary by individual. "
    "Always confirm with a licensed doctor or pharmacist before starting, "
    "stopping, or combining any medication."
)

# Reverse lookup: display name lower → class_name key
_NAME_TO_CLASS: Dict[str, str] = {
    info.get("name", "").strip().lower(): cls
    for cls, info in SKIN_DISEASE_DATABASE.items()
}

# ── Context regex ─────────────────────────────────────────────────────
_CTX_RE = re.compile(
    r"DETECTED:\s*(?P<disease>[^|]+?)\s*\|"
    r"\s*Severity:\s*(?P<severity>[^|]+?)\s*\|"
    r"\s*Confidence:\s*(?P<confidence>.+)$"
)

# ── Intent keyword groups ─────────────────────────────────────────────
_KW_MEDICINE   = {"medicine", "medication", "drug", "treat", "take", "use for",
                  "prescribe", "cure", "remedy", "tablet", "cream", "ointment",
                  "gel", "pill", "dosage", "dose", "apply", "suggest", "recommend",
                  "best medicine", "should i use", "good for", "what should i take"}
_KW_OVERVIEW   = {"what is", "tell me about", "explain", "describe", "overview",
                  "info", "information", "details", "about this", "disease"}
_KW_SYMPTOM    = {"symptom", "sign", "look like", "appear", "feel like",
                  "itch", "pain", "burning", "rash", "blister", "sore"}
_KW_CAUSE      = {"cause", "why", "reason", "trigger", "risk", "factor", "origin"}
_KW_CONTAGIOUS = {"contagious", "spread", "catch", "infectious", "transmit",
                  "pass on", "contact"}
_KW_PREVENT    = {"prevent", "avoid", "precaution", "protect", "stop", "reduce risk",
                  "hygiene"}
_KW_DOCTOR     = {"doctor", "hospital", "emergency", "urgent", "see a", "consult",
                  "when to", "visit", "serious"}
_KW_SEVERITY   = {"severe", "severity", "serious", "dangerous", "critical",
                  "mild", "moderate", "bad", "worse", "worsen"}
_KW_HOME       = {"home", "natural", "home remedy", "self", "without doctor",
                  "food", "diet", "eat", "drink", "lifestyle"}
_KW_SIDEFFECT  = {"side effect", "side-effect", "adverse", "reaction",
                  "safe", "safety", "danger", "risk of", "harm"}
_KW_PREGNANCY  = {"pregnant", "pregnancy", "breastfeed", "nursing", "baby",
                  "trimester", "fetal"}
_KW_CHILDREN   = {"child", "children", "kid", "infant", "baby", "age", "paediatric",
                  "pediatric"}


def _matches(text_lower: str, keywords: set) -> bool:
    return any(kw in text_lower for kw in keywords)


# ── Public helpers ────────────────────────────────────────────────────

def parse_short_context(
    ctx: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse 'DETECTED: X | Severity: Y | Confidence: Z' strings.
    Returns (disease_name, severity, confidence) or (None, None, None)."""
    if not ctx or ctx == "no_detection":
        return None, None, None
    m = _CTX_RE.search(ctx)
    if not m:
        return None, None, None
    return (
        m.group("disease").strip(),
        m.group("severity").strip(),
        m.group("confidence").strip(),
    )


def find_class_by_disease_name(disease_name: Optional[str]) -> Optional[str]:
    """Map a human-readable disease name back to its class_name DB key.
    Tries exact match, then substring match."""
    if not disease_name:
        return None
    key = disease_name.strip().lower()
    if key in _NAME_TO_CLASS:
        return _NAME_TO_CLASS[key]
    for name_lower, cls in _NAME_TO_CLASS.items():
        if name_lower and (name_lower in key or key in name_lower):
            return cls
    return None


def get_medicines_for_class(class_name: str) -> List[dict]:
    """Return enriched medicine dicts for a disease class (generic_name injected)."""
    names = DISEASE_MEDICATIONS.get(class_name, [])
    return [
        {"generic_name": n, **MEDICINE_DATABASE[n]}
        for n in names
        if n in MEDICINE_DATABASE
    ]


# ── Formatting helpers ────────────────────────────────────────────────

def _rx_label(rx) -> str:
    if rx is True:
        return "**Prescription required**"
    if rx is False:
        return "Available OTC (no prescription needed)"
    return str(rx)


def _format_medicine_list(meds: List[dict], limit: int = 8) -> str:
    """Compact bullet list: name, drug class, OTC/Rx status."""
    if not meds:
        return ""
    lines = []
    for med in meds[:limit]:
        rx = med.get("prescription_required")
        rx_tag = "🔴 Rx" if rx is True else "🟢 OTC"
        lines.append(
            f"• **{med['generic_name']}** ({med.get('drug_class', '—')})  {rx_tag}"
        )
    return "\n".join(lines)


def _format_medicine_detail(name: str, e: dict) -> str:
    """Full safety card for a single medicine."""
    rx = e.get("prescription_required")
    lines = [
        f"### 💊 {name}",
        f"**Drug class:** {e.get('drug_class', '—')}",
        f"**Used for:** {e.get('indications', '—')}",
        f"**Availability:** {_rx_label(rx)}",
        f"**Avoid if:** {e.get('contraindications', '—')}",
        f"**Side effects:** {e.get('side_effects', '—')}",
        f"**Pregnancy note:** {e.get('pregnancy_warning', '—')}",
        f"**Age restriction:** {e.get('age_limit', 'None stated')}",
        f"**Evidence basis:** {e.get('evidence_source', '—')}",
    ]
    return "\n\n".join(lines)


def _find_medicine_entry(text_lower: str) -> Optional[Tuple[str, dict]]:
    """Return (name, entry) if any medicine name appears in the user's message."""
    for name, entry in MEDICINE_DATABASE.items():
        if name.lower() in text_lower:
            return name, entry
    return None


_STOPWORDS = {"and", "the", "for", "with", "skin", "disease", "diseases", "other",
              "photos", "photo", "of", "or", "infection", "infections"}


def _find_disease_in_text(text_lower: str) -> Optional[str]:
    """Best-effort, fully-offline match of a disease/condition mentioned
    directly in the chat message (e.g. "medicine for eczema", "I think I
    have acne, what should I use?") against the local disease database.

    This lets DrugGPT suggest medicines straight from MEDICINE_DATABASE even
    when the person hasn't used the Detect tab or the symptom box — it never
    calls out to the network, it only reads SKIN_DISEASE_DATABASE / the
    medicine JSON already loaded in this module.
    """
    # 1) Full display-name match — handles short, single-concept names.
    for name_lower, cls in _NAME_TO_CLASS.items():
        if name_lower and name_lower in text_lower:
            return cls

    # 2) Word-level match — handles casual mentions of one distinctive word
    #    from a longer name, e.g. "eczema" inside "Eczema (Atopic Dermatitis)".
    #    Word boundaries avoid partial-word false hits.
    for name_lower, cls in _NAME_TO_CLASS.items():
        for word in re.findall(r"[a-z]+", name_lower):
            if len(word) > 3 and word not in _STOPWORDS and \
               re.search(rf"\b{re.escape(word)}\b", text_lower):
                return cls

    # 3) Light keyword-overlap fallback against each disease's symptom/cause
    #    text, in case the person describes the condition rather than naming
    #    it (e.g. "itchy red flaky patches"). Same idea as the Detect tab's
    #    analyze_symptoms(), kept local here to avoid a circular import.
    words = {w.strip(".,!?;:") for w in text_lower.split() if len(w) > 3}
    if not words:
        return None

    best_cls, best_score = None, 0
    for cls, info in SKIN_DISEASE_DATABASE.items():
        haystack = " ".join([info.get("symptoms", ""), info.get("causes", "")]).lower()
        hay_words = {w.strip(".,!?;:") for w in haystack.split() if len(w) > 3}
        score = len(words & hay_words)
        if score > best_score:
            best_cls, best_score = cls, score

    # Require at least 2 overlapping descriptive words to avoid wild guesses.
    return best_cls if best_score >= 2 else None


# ── Intent handlers ───────────────────────────────────────────────────

def _reply_medicine_detail(name: str, entry: dict) -> str:
    return _format_medicine_detail(name, entry) + f"\n\n_{SAFETY_DISCLAIMER}_"


def _reply_medicine_list(cls: str, info: dict) -> str:
    meds = get_medicines_for_class(cls)
    if not meds:
        note = NO_MEDICATION_NOTES.get(cls, "No standard OTC/topical medicines are listed for this condition.")
        return f"{note}\n\n_{SAFETY_DISCLAIMER}_"
    tip = f'Ask me "tell me about {meds[0]["generic_name"]}" for its full safety profile.'
    return (
        f"### 💊 Medicines commonly used for **{info['name']}**\n\n"
        f"{_format_medicine_list(meds)}\n\n"
        f"{tip}\n\n_{SAFETY_DISCLAIMER}_"
    )


def _reply_overview(cls: str, info: dict) -> str:
    meds = get_medicines_for_class(cls)
    med_section = (
        f"\n\n**Commonly used medicines:**\n{_format_medicine_list(meds)}"
        if meds
        else f"\n\n{NO_MEDICATION_NOTES.get(cls, '')}"
    )
    return (
        f"### {info['name']}  ·  {info.get('category', '')}\n\n"
        f"**Symptoms:** {info.get('symptoms', '—')}\n\n"
        f"**Causes:** {info.get('causes', '—')}\n\n"
        f"**Topical treatment:** {info.get('treatment_topical', '—')}\n\n"
        f"**Systemic treatment:** {info.get('treatment_systemic', '—')}"
        f"{med_section}\n\n"
        f"**When to see a doctor:** {info.get('doctor_visit', '—')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_symptoms(info: dict) -> str:
    return (
        f"### 🔍 Symptoms of **{info['name']}**\n\n"
        f"{info.get('symptoms', 'No symptom data available.')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_causes(info: dict) -> str:
    return (
        f"### 🧬 Causes & Risk Factors — **{info['name']}**\n\n"
        f"{info.get('causes', 'No cause data available.')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_contagious(cls: str, info: dict) -> str:
    contagious = info.get("contagious", False)
    verdict = (
        "⚠️ **Yes — this condition is contagious.** Avoid direct skin contact with others, "
        "do not share towels or clothing, and wash hands frequently until treated."
        if contagious
        else "✅ **No — this condition is not contagious.** It cannot spread from person to person."
    )
    return f"### 🦠 Contagious? — **{info['name']}**\n\n{verdict}\n\n_{SAFETY_DISCLAIMER}_"


def _reply_prevention(info: dict) -> str:
    return (
        f"### 🛡️ Prevention — **{info['name']}**\n\n"
        f"{info.get('prevention', 'No specific prevention data available.')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_doctor(info: dict) -> str:
    return (
        f"### 🏥 When to See a Doctor — **{info['name']}**\n\n"
        f"{info.get('doctor_visit', 'Consult a dermatologist if symptoms persist or worsen.')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_severity(info: dict) -> str:
    sev = info.get("severity", "Unknown")
    descriptions = {
        "None":     "No clinical risk — the condition is benign and self-limiting.",
        "Low":      "Low severity — manageable at home with OTC products in most cases.",
        "Medium":   "Moderate severity — medical evaluation is recommended.",
        "High":     "High severity — prompt medical attention is advised.",
        "Critical": "Critical — seek immediate medical care. Do not delay.",
        "Unknown":  "Severity not established. Consult a dermatologist for evaluation.",
    }
    return (
        f"### ⚡ Severity — **{info['name']}**\n\n"
        f"**Level:** {sev}\n\n"
        f"{descriptions.get(sev, 'See a doctor for evaluation.')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_home_remedies(info: dict) -> str:
    return (
        f"### 🌿 Home Care — **{info['name']}**\n\n"
        f"{info.get('home_remedies', 'No home remedy data available.')}\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


def _reply_side_effects(cls: str, info: dict) -> str:
    """List side effects for all medicines linked to this disease."""
    meds = get_medicines_for_class(cls)
    if not meds:
        return (
            f"No standard medicines are listed for **{info['name']}**, "
            f"so no side effect data is available here.\n\n_{SAFETY_DISCLAIMER}_"
        )
    lines = []
    for med in meds:
        se = med.get("side_effects", "—")
        lines.append(f"**{med['generic_name']}:** {se}")
    return (
        f"### ⚠️ Side Effects — medicines for **{info['name']}**\n\n"
        + "\n\n".join(lines)
        + f"\n\n_{SAFETY_DISCLAIMER}_"
    )


def _reply_pregnancy(cls: str, info: dict) -> str:
    meds = get_medicines_for_class(cls)
    if not meds:
        return (
            f"No standard medicines are listed for **{info['name']}**. "
            f"Always consult your OB-GYN or dermatologist during pregnancy.\n\n_{SAFETY_DISCLAIMER}_"
        )
    lines = []
    for med in meds:
        pw = med.get("pregnancy_warning", "—")
        lines.append(f"**{med['generic_name']}:** {pw}")
    return (
        f"### 🤰 Pregnancy Safety — medicines for **{info['name']}**\n\n"
        + "\n\n".join(lines)
        + f"\n\n_{SAFETY_DISCLAIMER}_"
    )


def _reply_children(cls: str, info: dict) -> str:
    meds = get_medicines_for_class(cls)
    if not meds:
        return (
            f"No standard medicines are listed for **{info['name']}**. "
            f"Consult a paediatrician before any treatment in children.\n\n_{SAFETY_DISCLAIMER}_"
        )
    lines = []
    for med in meds:
        al = med.get("age_limit", "Not specified")
        lines.append(f"**{med['generic_name']}:** {al}")
    return (
        f"### 👶 Age & Children — medicines for **{info['name']}**\n\n"
        + "\n\n".join(lines)
        + f"\n\n_{SAFETY_DISCLAIMER}_"
    )


def _reply_no_context() -> str:
    return (
        "I don't have an active condition loaded yet.\n\n"
        "👉 **Detect tab** — upload a skin photo and run analysis,\n"
        "👉 **Describe your symptoms** in the text box above and click *Analyze Symptoms*, **or**\n"
        "👉 Just tell me the condition right here, e.g. *\"what medicine for eczema?\"*\n\n"
        "Once a condition is set, ask me anything:\n"
        "• *What medicine should I use?*\n"
        "• *Is it contagious?*\n"
        "• *What causes it?*\n"
        "• *When should I see a doctor?*\n\n"
        f"_{SAFETY_DISCLAIMER}_"
    )


# ── Core offline reply engine ─────────────────────────────────────────

def _offline_reply(message: str, drug_context: Optional[str]) -> str:
    """
    Route the user's message to the right handler using keyword intent detection.
    All data comes exclusively from MEDICINE_DATABASE and SKIN_DISEASE_DATABASE.
    """
    text       = (message or "").strip()
    text_lower = text.lower()

    # ── 1. Named medicine lookup (highest priority) ──────────────────
    med_hit = _find_medicine_entry(text_lower)
    if med_hit:
        return _reply_medicine_detail(*med_hit)

    # ── 2. Resolve active disease context ───────────────────────────
    disease_name, severity, confidence = parse_short_context(drug_context)
    cls  = find_class_by_disease_name(disease_name) if disease_name else None

    # No active Detect-tab / symptom-box context? Try to infer the disease
    # straight from what was typed, so DrugGPT can still suggest medicines
    # from MEDICINE_DATABASE without forcing a trip to another tab first.
    if not cls:
        cls = _find_disease_in_text(text_lower)

    info = get_disease_info(cls) if cls else None

    if not cls or not info:
        return _reply_no_context()

    # ── 3. Intent routing ────────────────────────────────────────────
    if _matches(text_lower, _KW_PREGNANCY):
        return _reply_pregnancy(cls, info)

    if _matches(text_lower, _KW_CHILDREN):
        return _reply_children(cls, info)

    if _matches(text_lower, _KW_SIDEFFECT):
        return _reply_side_effects(cls, info)

    if _matches(text_lower, _KW_CONTAGIOUS):
        return _reply_contagious(cls, info)

    if _matches(text_lower, _KW_PREVENT):
        return _reply_prevention(info)

    if _matches(text_lower, _KW_DOCTOR):
        return _reply_doctor(info)

    if _matches(text_lower, _KW_SEVERITY):
        return _reply_severity(info)

    if _matches(text_lower, _KW_HOME):
        return _reply_home_remedies(info)

    if _matches(text_lower, _KW_CAUSE):
        return _reply_causes(info)

    if _matches(text_lower, _KW_SYMPTOM):
        return _reply_symptoms(info)

    if _matches(text_lower, _KW_MEDICINE):
        return _reply_medicine_list(cls, info)

    if _matches(text_lower, _KW_OVERVIEW):
        return _reply_overview(cls, info)

    # ── 4. Catch-all: full overview for the active condition ─────────
    return _reply_overview(cls, info)


# ── Public entry point ────────────────────────────────────────────────

def get_druggpt_reply(
    message: str,
    history: list,
    drug_context: Optional[str],
) -> str:
    """
    Return DrugGPT's offline reply to `message`.

    Parameters
    ----------
    message      : the user's latest message
    history      : list of {"role": "user"|"assistant", "content": str} dicts
                   (kept for signature compatibility; not used in offline mode)
    drug_context : context string from detect/symptom tabs, e.g.
                   "DETECTED: Acne & Rosacea | Severity: Low | Confidence: 87.3%"
    """
    if not message or not message.strip():
        return "Please type a question and I'll do my best to help. 😊"
    return _offline_reply(message, drug_context)