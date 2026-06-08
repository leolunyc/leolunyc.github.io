"""
app.py — LinkedIn Jobs Scraper with Resume Matching
Streamlit webapp: enter companies + job title keywords, upload resume,
get back jobs scored ≥ threshold (default 75%).
"""

import io
import os
import re
import time
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LinkedIn Jobs Scraper",
    page_icon="💼",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

REQUEST_DELAY = 2.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Resume parsing ─────────────────────────────────────────────────────────────

def parse_resume(uploaded_file) -> str:
    data = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Could not parse PDF: {e}")
            return ""
    elif name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            st.error(f"Could not parse DOCX: {e}")
            return ""
    return data.decode("utf-8", errors="ignore")

# ── Company input parsing ──────────────────────────────────────────────────────

def _parse_company_input(raw: str) -> tuple:
    """
    Returns (display_name, company_id_or_None).
    Accepts 'Company Name' or 'Company Name|12345' (precision override).
    """
    if "|" in raw:
        parts = raw.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return raw.strip(), None

# ── Job scraping ───────────────────────────────────────────────────────────────

def _company_matches(card_company: str, target: str) -> bool:
    """Fuzzy company name match: normalize to alphanum, check equality or containment."""
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    c, t = norm(card_company), norm(target)
    return bool(c and t and (c == t or (len(t) >= 4 and (t in c or c in t))))


def _parse_cards(html: str, company_name: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    jobs = []
    for card in soup.select("li"):
        title_tag = card.select_one(".base-search-card__title")
        location_tag = card.select_one(".job-search-card__location")
        link_tag = card.select_one("a.base-card__full-link")
        subtitle_tag = card.select_one(".base-search-card__subtitle")
        if not title_tag:
            continue
        href = (link_tag.get("href") or "") if link_tag else ""
        link = href.split("?")[0] if href else ""
        try:
            job_id = href.split("/view/")[1].split("?")[0].strip() if "/view/" in href else link
        except IndexError:
            job_id = link
        jobs.append({
            "id": job_id,
            "company": company_name,
            "_card_company": subtitle_tag.get_text(strip=True) if subtitle_tag else "",
            "title": title_tag.get_text(strip=True),
            "location": location_tag.get_text(strip=True) if location_tag else "",
            "link": link,
        })
    return jobs


def fetch_jobs(
    company_name: str,
    company_id: Optional[str],
    keyword: str,
    session: requests.Session,
) -> list[dict]:
    """
    Two modes:
    - company_id provided → filter by f_C (precise, requires numeric ID)
    - no company_id → search '{company_name} {keyword}', post-filter by card subtitle
    """
    if company_id:
        url = (
            "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
            f"?keywords={quote_plus(keyword)}&f_C={company_id}&start=0"
        )
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            jobs = _parse_cards(resp.text, company_name)
            return [_strip_internal(j) for j in jobs]
        except Exception:
            return []
    else:
        combined = f"{company_name} {keyword}"
        url = (
            "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
            f"?keywords={quote_plus(combined)}&start=0"
        )
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            jobs = _parse_cards(resp.text, company_name)
            # Keep only cards whose subtitle matches the target company
            matched = [j for j in jobs if _company_matches(j["_card_company"], company_name)]
            return [_strip_internal(j) for j in matched]
        except Exception:
            return []


def _strip_internal(job: dict) -> dict:
    """Remove internal-only fields before returning."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def fetch_description(link: str, session: requests.Session) -> str:
    m = re.search(r"-(\d+)/?$", link.rstrip("/"))
    if not m:
        return ""
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{m.group(1)}"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        desc = soup.find("div", {"class": re.compile(r"description|details|content", re.I)})
        if desc:
            return desc.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)[:6000]
    except Exception:
        return ""

# ── Scoring ────────────────────────────────────────────────────────────────────

def score_claude(resume: str, job_desc: str, title: str, api_key: str) -> Optional[int]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    "You are a hiring expert. Score how well this candidate's resume matches "
                    "the job posting on a scale of 0-100 (100 = perfect match).\n\n"
                    f"Job Title: {title}\n\n"
                    f"Job Description:\n{job_desc[:3000]}\n\n"
                    f"Resume:\n{resume[:3000]}\n\n"
                    "Reply with ONLY a single integer between 0 and 100. No explanation."
                ),
            }],
        )
        m = re.search(r"\d+", msg.content[0].text.strip())
        return int(m.group()) if m else None
    except Exception:
        return None


# ── Domain-aware fallback scorer (ported from filter_by_resume.py) ────────────
# Title patterns → immediate base scores

_STRONG_KEEP = re.compile(
    r"strategic\s+partnerships|"
    r"fintech\s+partnerships|"
    r"product\s+partnerships|"
    r"partnerships\s+manager|"
    r"partnership\s+manager|"
    r"director.*partnerships|"
    r"business\s+development.*partner|"
    r"partner.*business\s+development|"
    r"chief\s+of\s+staff|"
    r"go.to.market|"
    r"gtm\b",
    re.I,
)

_MODERATE_KEEP = re.compile(
    r"\bpartnerships?\b(?!.*engineer|.*developer|.*architect)|"
    r"commercializ|"
    r"biz\s*dev\b",
    re.I,
)

_HARD_DELETE = re.compile(
    r"software\s+engineer|senior\s+engineer|lead\s+engineer|"
    r"principal\s+engineer|staff\s+engineer|security\s+engineer|"
    r"backend\s+engineer|full.stack\s+engineer|frontend\s+engineer|"
    r"cryptograph|lead\s+architect|site\s+reliability|"
    r"financial\s+crimes|fraud\s+control|fraud\s+specialist|"
    r"compliance\s+investigat|crypto\s+trade\s+support|"
    r"deputy\s+editor|senior\s+counsel\b|\bfull.stack\b",
    re.I,
)

_MODERATE_DELETE = re.compile(
    r"\bengineer\b|\bdeveloper\b|\bscientist\b|\barchitect\b|"
    r"\bauditor\b|\banalyst\b(?!.*strateg|.*business|.*market)|"
    r"\beditor\b|\bcounsel\b|\battorney\b",
    re.I,
)

_POS_TERMS = [
    "partnerships", "partnership", "business development", "go-to-market", "gtm",
    "strategic", "fintech", "digital assets", "crypto", "blockchain", "stablecoin",
    "defi", "tokenization", "pipeline", "revenue", "commercial", "stakeholder",
    "cross-functional", "product roadmap", "product strategy", "chief of staff",
    "fundraising", "venture", "investor", "deal", "negotiat", "integration partner",
    "enterprise", "institutional", "financial services", "payments", "biz ops",
    "sales collateral", "abm", "icp", "market opportunity", "growth strategy",
    "head of sales", "sales strategy", "enterprise sales", "client acquisition",
    "market expansion", "revenue growth",
]

_NEG_TERMS = [
    "golang", "rust", "java", "c++", "kubernetes", "docker", "aws lambda",
    "backend", "frontend", "cryptography", "encryption", "key management",
    "sre", "devops", "cicd", "ci/cd", "api development", "microservices",
    "software development lifecycle", "sdlc",
    "fraud investigation", "bsa/aml compliance", "regulatory examination",
    "legal advice", "litigation",
]


def score_domain(title: str, job_desc: str) -> int:
    """
    Domain-aware scorer for a fintech / strategic-partnerships resume.
    Identical logic to filter_by_resume.py → score_domain_match().

    Bucket → base:
      STRONG_KEEP title   → 85
      MODERATE_KEEP title → 76
      HARD_DELETE title   → 20  (overrides everything)
      MODERATE_DELETE     → 35
      Neutral             → 50 ± description signals
    Then: ±10 for description keyword hits, capped [0, 100].
    """
    text = (title + " " + job_desc[:3000]).lower()
    pos = sum(1 for t in _POS_TERMS if t in text)
    neg = sum(1 for t in _NEG_TERMS if t in text)

    if _HARD_DELETE.search(title):
        base = 20
    elif _STRONG_KEEP.search(title):
        base = 85 + min(10, pos * 2) - min(10, neg * 3)
    elif _MODERATE_KEEP.search(title):
        base = 76 + min(10, pos * 2) - min(10, neg * 3)
    elif _MODERATE_DELETE.search(title):
        base = 35 + min(5, pos) - min(10, neg * 3)
    else:
        base = 50 + (pos * 5) - (neg * 8)

    return max(0, min(100, base))

# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    companies_raw: list[str],
    titles: list[str],
    resume_text: str,
    li_at: str,
    api_key: str,
    threshold: int,
) -> pd.DataFrame:

    session = requests.Session()
    session.headers.update(HEADERS)
    if li_at:
        session.cookies.set("li_at", li_at.strip(), domain=".linkedin.com")

    # Parse company list — name-only or name|id pairs
    companies = [_parse_company_input(r) for r in companies_raw]
    id_count = sum(1 for _, cid in companies if cid)
    if id_count:
        st.info(f"{id_count} / {len(companies)} companies have explicit LinkedIn IDs (precise mode). The rest use keyword search.")

    # ── Step 1: Fetch job listings ────────────────────────────────────────
    total = len(companies) * len(titles)
    st.markdown(f"**Step 1 / 2 — Fetching jobs** ({len(companies)} companies × {len(titles)} keywords = up to {total} searches)")
    bar2 = st.progress(0)
    note2 = st.empty()

    all_jobs: list[dict] = []
    seen_links: set[str] = set()
    call = 0

    for name, cid in companies:
        for kw in titles:
            mode = "ID filter" if cid else "keyword search"
            note2.caption(f'[{mode}] "{kw}" at {name}...')
            jobs = fetch_jobs(name, cid, kw, session)
            for job in jobs:
                link = job.get("link", "")
                if link and link not in seen_links:
                    seen_links.add(link)
                    all_jobs.append(job)
            call += 1
            bar2.progress(call / total)
            time.sleep(REQUEST_DELAY)

    note2.empty()
    bar2.empty()

    if not all_jobs:
        st.warning("No jobs found. Try different keywords or companies.")
        return pd.DataFrame()

    st.success(f"Found {len(all_jobs)} unique job listings.")

    # ── Step 2: Fetch descriptions + score ───────────────────────────────
    use_claude = bool(api_key)
    method = "Claude AI" if use_claude else "domain pattern matching"
    st.markdown(f"**Step 2 / 2 — Scoring jobs** using {method}")
    bar3 = st.progress(0)
    note3 = st.empty()

    rows: list[dict] = []

    for i, job in enumerate(all_jobs):
        note3.caption(f"Scoring {i + 1}/{len(all_jobs)}: {job['title'][:55]}…")
        desc = fetch_description(job["link"], session)
        time.sleep(REQUEST_DELAY)

        if use_claude:
            score = score_claude(resume_text, desc, job["title"], api_key)
            if score is None:
                score = score_domain(job["title"], desc)
        else:
            score = score_domain(job["title"], desc)

        rows.append({
            "Score": score,
            "Title": job["title"],
            "Company": job["company"],
            "Location": job["location"],
            "Link": job["link"],
        })
        bar3.progress((i + 1) / len(all_jobs))

    note3.empty()
    bar3.empty()

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return df

# ── Results display ────────────────────────────────────────────────────────────

def show_results(df: pd.DataFrame, threshold: int) -> None:
    st.divider()
    st.subheader("Results")

    if df.empty:
        st.info("No results.")
        return

    matched = df[df["Score"] >= threshold]

    c1, c2, c3 = st.columns(3)
    c1.metric("Matching Jobs", len(matched))
    c2.metric("Total Found", len(df))
    c3.metric("Threshold", f"{threshold}%")

    if matched.empty:
        st.info(f"No jobs scored ≥ {threshold}%. Lower the threshold in Settings or add more companies.")
        return

    st.dataframe(
        matched,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%d%%"
            ),
            "Link": st.column_config.LinkColumn("Link", display_text="Open ↗"),
        },
        height=min(600, 38 * (len(matched) + 1) + 40),
    )

    st.download_button(
        "Download CSV",
        data=matched.to_csv(index=False),
        file_name="matching_jobs.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("💼 LinkedIn Jobs Scraper")
    st.caption("Enter target companies and job title keywords, upload your resume — see only the jobs that match.")

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        li_at = st.text_input(
            "LinkedIn `li_at` Cookie",
            type="password",
            help=(
                "Paste your LinkedIn session cookie to unlock companies that block "
                "guest access (Wells Fargo, JPMorgan, etc.). "
                "Find it: DevTools → Application → Cookies → linkedin.com → li_at."
            ),
        )

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help=(
                "Uses Claude Haiku to semantically score each job against your resume. "
                "Highly recommended — much more accurate than keyword matching. "
                "Falls back to TF-IDF if not provided."
            ),
        )

        threshold = st.slider(
            "Match Score Threshold",
            min_value=50, max_value=95, value=75, step=5,
            help="Only jobs at or above this score are shown in results.",
        )

        st.divider()
        st.caption(
            "**Precision mode (optional):** Append a LinkedIn company ID to get "
            "exact results: `Stripe|40950`. "
            "To find an ID: go to the company's LinkedIn page, open DevTools → "
            "Network, and look for `f_C=` in any jobs request."
        )

    # ── Inputs ─────────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Target Companies")
        st.caption("One per line, up to 25. Just the name is fine. Optional precision mode: `Stripe|40950`")
        companies_raw_text = st.text_area(
            "companies",
            height=340,
            placeholder="Stripe\nCoinbase\nRipple\nCircle\nAnchorage Digital\nPaxos\n...",
            label_visibility="collapsed",
        )
        companies_list = [c.strip() for c in companies_raw_text.strip().splitlines() if c.strip()][:25]
        if companies_list:
            st.caption(f"{len(companies_list)} / 25 companies")

    with right:
        st.subheader("Job Title Keywords")
        st.caption("Up to 5. Each keyword is searched at every company.")
        kws: list[str] = []
        placeholders = [
            "Partnerships Manager",
            "Business Development",
            "Strategic Partnerships",
            "Head of Partnerships",
            "GTM / Go-to-Market",
        ]
        for i in range(5):
            t = st.text_input(f"Keyword {i + 1}", key=f"kw_{i}", placeholder=placeholders[i])
            if t.strip():
                kws.append(t.strip())
        if kws:
            st.caption(f"{len(kws)} / 5 keywords")

    st.subheader("Resume")
    resume_file = st.file_uploader(
        "Upload your resume (PDF or DOCX)",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )
    resume_text = ""
    if resume_file:
        resume_text = parse_resume(resume_file)
        if resume_text:
            st.success(f"Resume parsed — {len(resume_text):,} characters extracted.")

    # ── Run ─────────────────────────────────────────────────────────────────
    missing = []
    if not companies_list:
        missing.append("target companies")
    if not kws:
        missing.append("job title keywords")
    if not resume_text:
        missing.append("resume")

    if missing:
        st.info(f"To search, please provide: {', '.join(missing)}.")

    if st.button(
        "Search Jobs",
        disabled=bool(missing),
        type="primary",
        use_container_width=True,
    ):
        st.divider()
        df = run_pipeline(companies_list, kws, resume_text, li_at, api_key, threshold)
        st.session_state["results_df"] = df
        st.session_state["results_threshold"] = threshold

    if "results_df" in st.session_state:
        show_results(
            st.session_state["results_df"],
            st.session_state.get("results_threshold", threshold),
        )


if __name__ == "__main__":
    main()
