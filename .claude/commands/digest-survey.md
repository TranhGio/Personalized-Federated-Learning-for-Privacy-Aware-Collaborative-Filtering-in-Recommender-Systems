# Digest Survey Paper into Structured Knowledge

You are a research assistant specializing in **Federated Learning**, **Recommender Systems**, and **Privacy-Preserving ML**. Your task is to read a **survey paper** and produce a structured digest optimized for future AI-agent consumption.

<thesis_context>
- **Title**: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
- **Baseline**: BPR-MF (Bayesian Personalized Ranking Matrix Factorization)
- **Dataset**: MovieLens-1M
- **FL Framework**: Flower
- **Metrics**: NDCG@K, HitRate@K (K = 5, 10, 20)
- **Core themes**: user-centric personalization, privacy (DP), non-IID data heterogeneity, communication efficiency
</thesis_context>

<survey_vs_method>
This prompt is for SURVEY papers. Surveys differ from method papers in that they:
- Cover dozens of methods rather than proposing one
- Provide taxonomies and classification frameworks
- Contain comparative tables across many papers
- Identify research gaps and future directions
- May formalize general optimization objectives that unify multiple methods

Do NOT try to force a survey into the single-method template.
Instead, extract the survey's STRUCTURE as a knowledge map.
</survey_vs_method>

<task>
1. Read the survey paper at the path provided: $ARGUMENTS
2. Extract ALL technical details following the output format below
3. Write the digest to `Papers/digested/<paper_id>.md` where `<paper_id>` is `<first_author_lastname>_<year>_survey_<keyword>` in lowercase (e.g., `zhang_2025_survey_personalized_fedrec`)
4. Update `Papers/digested/_INDEX.md`:
   - Add the survey entry with a `[SURVEY]` tag at the top of the relevant topic section
   - Surveys should appear BEFORE individual method papers in each section since they provide overview context
</task>

<reading_strategy>
- First: extract text to get title, abstract, table of contents / section headings — this gives you the taxonomy structure
- Then: focus on taxonomy diagrams and classification figures — rasterize these pages as they contain the paper's core organizational framework
- Then: extract ALL comparative tables — these are the most valuable parts of a survey for implementation planning
- Then: read the sections on challenges, solutions, and future directions
- Then: extract formal definitions and unified optimization objectives if provided
- Finally: collect the reference list, focusing on papers that appear repeatedly across multiple tables or are highlighted as seminal/pioneering
</reading_strategy>

<output_format>
Write the digest as a Markdown file with EXACTLY this structure:

```markdown
# [SURVEY] [Full Paper Title]

- **Authors**: [First Author, Second Author, et al.]
- **Venue**: [Conference/Journal, Year]
- **Paper ID**: [paper_id as defined above]
- **Tags**: [comma-separated tags, e.g., #survey, #federated-recommender, #personalization]
- **Scope**: [One sentence: what does this survey cover and what is its unique angle?]

---

## 1. Survey Landscape

### 1.1 What This Survey Covers
[2-3 sentences describing the survey's scope, what distinguishes it from prior surveys]

### 1.2 Taxonomy Overview
[Reproduce the survey's main classification hierarchy as a tree structure. This is the MOST IMPORTANT section — it serves as a navigation map.]

```
Root Topic
├── Branch A
│   ├── Sub-branch A1: [1-line description]
│   │   └── Representative papers: [cite 2-3]
│   └── Sub-branch A2: [1-line description]
│       └── Representative papers: [cite 2-3]
├── Branch B
│   ├── Sub-branch B1
│   ...
```

### 1.3 Key Figures and Diagrams
[Describe the most important figures in the survey — architecture diagrams, taxonomy trees, framework comparisons. Reference page numbers so the agent can rasterize them if needed later.]

- **Figure X (p.Y)**: [What it shows and why it matters]
- **Figure Z (p.W)**: [What it shows and why it matters]

---

## 2. Formal Definitions & Unified Objectives

[Many surveys provide formal definitions that unify the methods they review. Extract ALL of these — they are extremely valuable for understanding how different methods relate to each other.]

### Definition: [Name]
[Quote or paraphrase the formal definition]

### Unified Optimization Objective
[If the survey provides a general formulation that multiple methods instantiate, write it here in LaTeX]

$$
[unified objective]
$$

Where:
- $symbol$: [meaning]
- ...

### Objective Variants
[How different methods instantiate this general objective. Use a table:]

| Method | How it instantiates the objective | Key difference |
|---|---|---|
| ... | ... | ... |

---

## 3. Method Catalog

[This is the core reference section. For EACH method category in the taxonomy, create a structured entry. Focus on methods relevant to the thesis.]

### 3.1 [Category Name from Taxonomy]

**Overview**: [2-3 sentences on this category's approach]

**Methods Table**:
| Method | Year | Key Technique | Task | Datasets | Metrics | Code Available? |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

**Key Equations** (if the survey provides them for this category):
- [Method A]: $equation$
- [Method B]: $equation$

**Comparative Results** (if the survey provides cross-method comparison):
| Method | [Metric 1] | [Metric 2] | Dataset |
|---|---|---|---|
| ... | ... | ... | ... |

[Repeat 3.1 pattern for each major category: 3.2, 3.3, etc.]

---

## 4. Challenges & Solutions Framework

[Surveys often identify open challenges and map them to solutions. Extract this mapping completely.]

| Challenge | Description | Proposed Solutions | Relevant Methods |
|---|---|---|---|
| C1: [name] | [1-2 sentences] | S1: [approach] | [methods] |
| ... | ... | S2: [approach] | [methods] |

### Challenge Formulations
[If the survey formalizes challenges as optimization problems, extract each one]

**Challenge: [Name]**
$$
[formulation]
$$
Where: [explain symbols]

---

## 5. Research Gaps & Future Directions

[Extract ALL future directions mentioned. Rate each by relevance to thesis.]

| Future Direction | Survey's Description | Relevance to My Thesis | Why |
|---|---|---|---|
| [Direction 1] | [1-2 sentences] | 🔴 High / 🟡 Medium / 🟢 Low | [specific reason] |
| [Direction 2] | ... | ... | ... |

---

## 6. Key References Extracted

[From the survey's references, identify papers that are most cited within the survey OR most relevant to the thesis. Organize by priority.]

### Tier 1 — Must-read (directly relevant to thesis)
| Paper | Year | Why important | Already in our KB? |
|---|---|---|---|
| [Title] | [Year] | [reason] | Yes/No |

### Tier 2 — Should-read (provides useful techniques or baselines)
| Paper | Year | Why important | Already in our KB? |
|---|---|---|---|
| ... | ... | ... | ... |

### Tier 3 — Nice-to-know (broader context)
| Paper | Year | Why important |
|---|---|---|
| ... | ... | ... |

---

## 7. Connections to My Thesis

### 7.1 Directly Applicable Taxonomy Mappings
[Where does my thesis fit in this survey's taxonomy? Which branch? What are the closest existing methods?]

### 7.2 Methods to Prioritize for Implementation
[Based on this survey's comparative analysis, which specific methods should I implement or compare against? Why?]

### 7.3 Gaps My Thesis Can Address
[Which research gaps identified by this survey does my thesis proposal directly address? Be specific.]

### 7.4 Optimization Objectives to Build Upon
[Which formal objectives from this survey can serve as the starting point for my thesis formulation?]
```
</output_format>

<quality_checklist>
Before saving the digest, verify:
- [ ] Taxonomy tree is COMPLETE — no branches omitted
- [ ] ALL comparative tables from the survey are extracted (these are the highest-value content)
- [ ] Every method table includes the "Code Available?" column (critical for reproducibility)
- [ ] Formal definitions and unified objectives use proper LaTeX
- [ ] Future directions are ALL captured with relevance ratings
- [ ] "Key References Extracted" cross-references with existing papers in `_INDEX.md`
- [ ] Section 7 (Connections to My Thesis) is specific, not generic
- [ ] Key figures are documented with page numbers for future rasterization
</quality_checklist>

<important_rules>
- Survey digests are LONGER than method paper digests — this is expected and acceptable
- Prioritize TABLES and TAXONOMIES over prose — these are what make surveys uniquely valuable
- When extracting method catalogs, ALWAYS note which methods use MovieLens, BPR, MF, or implicit feedback
- Flag any method that combines personalization + FL + recommender system — these are the closest related works
- Use consistent notation: $w$ for global model, $\theta_i$ for client $i$'s personalized model, $\phi_u$ for personalized parameters, $\mathcal{D}_i$ for client $i$'s local dataset
- If the survey references methods already in our knowledge base, note the cross-reference explicitly
- For very long surveys (>20 pages), it's acceptable to summarize less-relevant categories briefly while giving full treatment to categories relevant to the thesis
</important_rules>
