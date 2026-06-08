"""20 hard-coded AI/ML passages for the Graph RAG benchmark.

Deliberately split so no single chunk answers any of the 5 multi-hop test questions.
Each passage carries an 'entities' list — the knowledge-graph node names that appear
in its text. These are stored in the Qdrant payload at index time so the graph leg
can find seed entities without an extra NER call.
"""
from __future__ import annotations

PASSAGES: list[dict] = [
    # ── Q5 chain: Sam Altman → OpenAI → GPT-4 → RLHF ──────────────────────
    {
        "id": "altman_ceo",
        "text": (
            "Sam Altman serves as CEO of OpenAI. "
            "He previously led Y Combinator as its president before joining OpenAI full-time."
        ),
        "entities": ["Sam Altman", "OpenAI"],
    },
    {
        "id": "gpt4",
        "text": (
            "GPT-4 is OpenAI's most capable language model, released in March 2023. "
            "It demonstrated human-level performance on a range of professional and academic benchmarks."
        ),
        "entities": ["GPT-4", "OpenAI"],
    },
    {
        "id": "rlhf",
        "text": (
            "OpenAI's language models are trained using Reinforcement Learning from Human Feedback (RLHF). "
            "This technique collects human preference data to fine-tune model behavior toward helpfulness and safety."
        ),
        "entities": ["OpenAI", "RLHF"],
    },
    # ── Q1 chain: Dario left OpenAI → Anthropic → Constitutional AI ─────────
    {
        "id": "dario_left",
        "text": (
            "Dario Amodei and his sister Daniela Amodei, along with several colleagues, "
            "departed from OpenAI in 2021 to found a new AI safety company."
        ),
        "entities": ["Dario Amodei", "Daniela Amodei", "OpenAI"],
    },
    {
        "id": "anthropic_founded",
        "text": (
            "Anthropic was incorporated in 2021 by Dario Amodei, Daniela Amodei, "
            "and other former OpenAI researchers. The company focuses on AI safety and interpretability research."
        ),
        "entities": ["Anthropic", "Dario Amodei", "Daniela Amodei"],
    },
    {
        "id": "constitutional_ai",
        "text": (
            "Constitutional AI is Anthropic's primary alignment technique. "
            "It uses a written set of principles to guide the model to critique and revise "
            "its own outputs, reducing harmful and dishonest behavior without relying solely on human labels."
        ),
        "entities": ["Constitutional AI", "Anthropic"],
    },
    {
        "id": "claude3",
        "text": (
            "Claude 3 Opus is Anthropic's flagship language model. "
            "It achieves top scores on benchmarks including MMLU and HumanEval, "
            "surpassing GPT-4 on several reasoning tasks."
        ),
        "entities": ["Claude 3", "Anthropic", "GPT-4"],
    },
    # ── Q2 chain: Demis → DeepMind → AlphaFold 2 → Protein Data Bank ────────
    {
        "id": "hassabis_deepmind",
        "text": (
            "Demis Hassabis co-founded DeepMind in 2010 in London. "
            "He holds a PhD in cognitive neuroscience from University College London "
            "and continues to lead DeepMind as its CEO."
        ),
        "entities": ["Demis Hassabis", "DeepMind"],
    },
    {
        "id": "alphafold",
        "text": (
            "DeepMind developed AlphaFold 2, which achieved atomic-accuracy prediction of protein "
            "3D structures. The system solved a 50-year-old grand challenge in biology "
            "and was published in Nature in 2021."
        ),
        "entities": ["DeepMind", "AlphaFold 2"],
    },
    {
        "id": "alphafold_data",
        "text": (
            "AlphaFold 2 was trained primarily on protein sequences from the UniProt database "
            "and experimentally determined structures from the Protein Data Bank (PDB). "
            "The PDB contains over 200,000 experimentally resolved structures."
        ),
        "entities": ["AlphaFold 2", "UniProt", "Protein Data Bank"],
    },
    {
        "id": "deepmind_google",
        "text": (
            "Google acquired DeepMind in 2014 for approximately $500 million, "
            "making it a wholly owned subsidiary of Alphabet Inc."
        ),
        "entities": ["Google", "DeepMind"],
    },
    # ── Q4 chain: Hinton → Google Brain → merger → Google DeepMind → Gemini ─
    {
        "id": "hinton_google",
        "text": (
            "Geoffrey Hinton joined Google Brain in 2013 and worked there for a decade. "
            "He departed in May 2023 to speak freely about the potential risks of artificial intelligence."
        ),
        "entities": ["Geoffrey Hinton", "Google Brain"],
    },
    {
        "id": "brain_deepmind_merger",
        "text": (
            "Google DeepMind was formed in April 2023 when Google Brain and DeepMind merged "
            "into a single unified AI research organization under Alphabet."
        ),
        "entities": ["Google DeepMind", "Google Brain", "DeepMind"],
    },
    {
        "id": "gemini",
        "text": (
            "Gemini Ultra is the first frontier model released by Google DeepMind "
            "following the merger of Google Brain and DeepMind. "
            "It uses a multimodal Transformer architecture."
        ),
        "entities": ["Gemini Ultra", "Google DeepMind", "Transformer"],
    },
    # ── Q3 chain: Vaswani → Transformer → LLaMA 2 → Common Crawl ────────────
    {
        "id": "vaswani",
        "text": (
            "The Transformer architecture was introduced in the paper "
            "'Attention is All You Need' by Vaswani et al. at Google Brain in 2017. "
            "It replaces recurrence with self-attention mechanisms, enabling parallelism."
        ),
        "entities": ["Transformer", "Vaswani et al.", "Google Brain"],
    },
    {
        "id": "llama2_arch",
        "text": (
            "LLaMA 2 uses a Transformer architecture enhanced with grouped-query attention "
            "and rotary positional embeddings (RoPE) for improved inference efficiency."
        ),
        "entities": ["LLaMA 2", "Transformer"],
    },
    {
        "id": "llama2_data",
        "text": (
            "LLaMA 2's pretraining corpus is sourced primarily from Common Crawl, "
            "supplemented by curated high-quality sources. "
            "The total dataset is approximately 2 trillion tokens."
        ),
        "entities": ["LLaMA 2", "Common Crawl"],
    },
    # ── Supporting passages ──────────────────────────────────────────────────
    {
        "id": "meta_ai",
        "text": (
            "Meta AI is the artificial intelligence research organization within Meta Platforms. "
            "Yann LeCun serves as Chief AI Scientist and advocates for self-supervised learning approaches."
        ),
        "entities": ["Meta AI", "Yann LeCun"],
    },
    {
        "id": "llama2_release",
        "text": (
            "LLaMA 2 was released by Meta AI in July 2023 as an open-weight model "
            "available under a custom commercial license."
        ),
        "entities": ["LLaMA 2", "Meta AI"],
    },
    {
        "id": "common_crawl",
        "text": (
            "Common Crawl is a non-profit organization that has been archiving the web since 2008. "
            "It provides petabytes of raw web data freely available for research and commercial use."
        ),
        "entities": ["Common Crawl"],
    },
]
