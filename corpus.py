"""
RAG Knowledge Assistant — Document Corpus
Mixed HR/Policy + Technical/ML documents for realistic demo.
In production: replace with PDF/DOCX loaders (see loaders.py)
"""

# ─────────────────────────────────────────────────────────────
# HR & COMPANY POLICY DOCUMENTS
# ─────────────────────────────────────────────────────────────

HR_DOCS = [
    {
        "id": "hr_001",
        "title": "Employee Leave Policy 2024",
        "category": "HR",
        "content": """
        ANNUAL LEAVE POLICY — TechCorp Inc.

        All full-time employees are entitled to 25 days of paid annual leave per calendar year.
        Part-time employees receive leave on a pro-rata basis calculated from their contracted hours.
        Leave must be requested at least 2 weeks in advance via the HR portal, except in cases of
        emergency. A maximum of 10 unused leave days may be carried forward to the following year.
        Any remaining balance beyond 10 days is forfeited on December 31st.

        SICK LEAVE
        Employees are entitled to 10 days paid sick leave per year. Medical certificates are
        required for absences exceeding 3 consecutive days. Sick leave does not carry forward
        and resets on January 1st each year.

        PARENTAL LEAVE
        Primary caregivers are entitled to 16 weeks of fully paid parental leave.
        Secondary caregivers receive 4 weeks of fully paid leave. Parental leave must be
        taken within 12 months of the child's birth or adoption date. Employees must notify
        HR at least 8 weeks before the intended leave start date.

        BEREAVEMENT LEAVE
        Employees are granted 5 days paid bereavement leave for the loss of an immediate
        family member (spouse, child, parent, sibling). 3 days are provided for extended
        family (grandparents, in-laws). Additional unpaid leave may be requested.
        """
    },
    {
        "id": "hr_002",
        "title": "Remote Work & Hybrid Policy",
        "category": "HR",
        "content": """
        REMOTE WORK POLICY — TechCorp Inc.

        TechCorp operates a hybrid-first model. Employees in engineering, data, and product
        roles may work remotely up to 3 days per week. All employees are expected to be
        present in the office on Tuesdays and Thursdays (core collaboration days).

        HOME OFFICE SETUP
        The company provides a one-time home office allowance of $1,500 for full-time employees
        who work remotely more than 2 days per week. This covers equipment such as monitors,
        keyboards, and ergonomic furniture. Receipts must be submitted within 60 days of purchase.

        INTERNET REIMBURSEMENT
        Employees working remotely are eligible for internet reimbursement of up to $60/month.
        Claims are submitted monthly via the expenses portal with proof of internet bill.

        SECURITY REQUIREMENTS
        Remote employees must use company-approved VPN at all times when accessing internal
        systems. Personal devices must have MDM (Mobile Device Management) software installed.
        Working from public Wi-Fi without VPN is strictly prohibited.

        INTERNATIONAL REMOTE WORK
        Employees wishing to work from another country must obtain approval from their manager
        and HR at least 4 weeks in advance. International remote work is limited to 30 days
        per calendar year and may have tax implications the employee is responsible for.
        """
    },
    {
        "id": "hr_003",
        "title": "Performance Review Process",
        "category": "HR",
        "content": """
        PERFORMANCE REVIEW CYCLE — TechCorp Inc.

        TechCorp conducts formal performance reviews twice per year: mid-year (June) and
        end-of-year (December). Reviews use a calibrated rating scale: Exceptional (5),
        Exceeds Expectations (4), Meets Expectations (3), Needs Improvement (2), Unsatisfactory (1).

        The review process involves three components:
        1. Self-assessment: Employee completes self-evaluation 2 weeks before review meeting
        2. Manager assessment: Direct manager provides ratings and written feedback
        3. 360 feedback: Optional peer feedback collected for senior roles (L5 and above)

        PROMOTION CRITERIA
        Promotion decisions are made at the end-of-year review. To be eligible, employees
        must have a minimum rating of 4 (Exceeds Expectations) for two consecutive review
        cycles. Promotions also require manager nomination and calibration committee approval.

        PERFORMANCE IMPROVEMENT PLANS
        Employees receiving a rating of 2 or below are placed on a 90-day Performance
        Improvement Plan (PIP). PIPs include specific measurable goals, weekly check-ins
        with the manager, and HR oversight. Failure to meet PIP goals may result in termination.

        COMPENSATION REVIEW
        Merit increases are tied to performance ratings. Exceptional performers receive
        8-12% salary increase. Exceeds Expectations: 5-7%. Meets Expectations: 2-4%.
        Needs Improvement: 0%. Increases are effective February 1st following the review.
        """
    },
    {
        "id": "hr_004",
        "title": "Code of Conduct & Ethics",
        "category": "HR",
        "content": """
        CODE OF CONDUCT — TechCorp Inc.

        CONFLICTS OF INTEREST
        Employees must disclose any potential conflicts of interest to their manager and HR.
        This includes outside employment, investments in competitors, or personal relationships
        with vendors. Undisclosed conflicts may result in disciplinary action.

        DATA PRIVACY
        All employees handling customer data must complete annual GDPR and data privacy
        training. Customer data must never be stored on personal devices or shared externally
        without explicit legal and security approval.

        ANTI-HARASSMENT POLICY
        TechCorp maintains a zero-tolerance policy for harassment, discrimination, or
        bullying of any kind. This includes verbal, written, and digital communication.
        Complaints should be reported to HR or via the anonymous ethics hotline at
        1-800-ETHICS-1. All reports are investigated within 10 business days.

        SOCIAL MEDIA
        Employees may not share confidential company information, unreleased product details,
        or customer data on social media. Employees representing TechCorp publicly must
        include the disclaimer: "Views are my own."

        GIFT POLICY
        Employees may not accept gifts valued above $50 from vendors, clients, or partners.
        All gifts above $25 must be declared in the gift registry in the HR portal.
        """
    },
    {
        "id": "hr_005",
        "title": "Compensation & Benefits Overview",
        "category": "HR",
        "content": """
        COMPENSATION & BENEFITS — TechCorp Inc.

        SALARY BANDS (2024)
        L1 (Junior): $65,000 - $85,000
        L2 (Mid): $85,000 - $115,000
        L3 (Senior): $115,000 - $155,000
        L4 (Staff): $155,000 - $210,000
        L5 (Principal): $210,000 - $280,000

        EQUITY
        All full-time employees receive RSU (Restricted Stock Unit) grants vesting over 4 years
        with a 1-year cliff. Grant sizes vary by level and are refreshed annually based on
        performance ratings.

        HEALTH INSURANCE
        TechCorp covers 100% of employee premiums for medical, dental, and vision insurance.
        Dependents can be added at 50% company contribution. Open enrollment is in November.

        401K
        The company matches 401K contributions up to 4% of base salary. Matching is immediate
        with no vesting period. The plan is managed through Fidelity.

        LEARNING & DEVELOPMENT
        Each employee has a $2,000 annual learning budget for courses, books, conferences,
        and certifications. Unused budget does not carry over. Requests are approved by managers.

        WELLNESS
        Employees receive a $500 annual wellness stipend for gym memberships, fitness equipment,
        therapy, or meditation apps. Claims submitted via the benefits portal.
        """
    },
]

# ─────────────────────────────────────────────────────────────
# TECHNICAL / ML DOCUMENTS
# ─────────────────────────────────────────────────────────────

TECH_DOCS = [
    {
        "id": "tech_001",
        "title": "Introduction to Transformer Architecture",
        "category": "Technical",
        "content": """
        TRANSFORMER ARCHITECTURE

        The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
        revolutionized natural language processing by replacing recurrent networks with self-attention
        mechanisms. Unlike RNNs, Transformers process all tokens in parallel, enabling efficient
        training on large datasets.

        SELF-ATTENTION MECHANISM
        Self-attention allows each token to attend to every other token in the sequence.
        For each token, three vectors are computed: Query (Q), Key (K), and Value (V).
        Attention scores are calculated as: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        where d_k is the dimension of the key vectors, used for scaling to prevent vanishing gradients.

        MULTI-HEAD ATTENTION
        Multi-head attention runs self-attention in parallel across h heads, each learning
        different representation subspaces. The outputs are concatenated and linearly projected.
        This allows the model to capture relationships at different scales simultaneously.

        POSITIONAL ENCODING
        Since Transformers have no inherent notion of sequence order, positional encodings
        are added to input embeddings. The original paper uses sinusoidal encodings.
        Modern models (BERT, GPT) use learned positional embeddings instead.

        FEED-FORWARD LAYERS
        Each Transformer block contains a position-wise feed-forward network applied
        independently to each token: FFN(x) = max(0, xW1 + b1)W2 + b2
        This adds non-linear transformation capacity beyond what attention provides.

        LAYER NORMALIZATION
        Residual connections and layer normalization are applied after each sub-layer.
        This stabilizes training of deep networks: Output = LayerNorm(x + SubLayer(x))
        """
    },
    {
        "id": "tech_002",
        "title": "RAG: Retrieval-Augmented Generation",
        "category": "Technical",
        "content": """
        RETRIEVAL-AUGMENTED GENERATION (RAG)

        RAG, introduced by Lewis et al. (2020), combines parametric memory (LLM weights)
        with non-parametric memory (retrieved documents) to ground language model outputs
        in factual external knowledge. This reduces hallucinations and enables knowledge
        updates without retraining.

        RAG PIPELINE COMPONENTS
        1. Document Store: A corpus of documents (PDFs, wikis, databases) split into chunks
        2. Embedding Model: Converts text to dense vector representations
        3. Vector Database: Stores and indexes embeddings for fast similarity search
        4. Retriever: Queries the vector database with the user question embedding
        5. Generator: LLM that receives retrieved context + question to produce grounded answers

        CHUNKING STRATEGIES
        Documents must be split into chunks before embedding. Common strategies:
        - Fixed-size chunking: Split every N tokens with M token overlap
        - Sentence-aware chunking: Split on sentence boundaries within size limits
        - Semantic chunking: Group sentences by semantic similarity (cosine distance)
        - Recursive chunking: Hierarchical splitting (paragraph → sentence → word)
        Overlap (typically 10-20% of chunk size) ensures context isn't lost at boundaries.

        RETRIEVAL METHODS
        - Dense retrieval: Embed query and find nearest neighbors (cosine/dot product)
        - Sparse retrieval: BM25 keyword matching (fast, interpretable)
        - Hybrid retrieval: Combine dense + sparse with reciprocal rank fusion
        - Re-ranking: Use a cross-encoder to re-score retrieved candidates

        HALLUCINATION REDUCTION
        RAG reduces hallucinations because the LLM is prompted to answer ONLY from
        retrieved context. Without RAG, LLMs rely on training data which may be outdated,
        wrong, or confidently fabricated (hallucinated). With RAG, answers are grounded
        in retrieved evidence that can be cited and verified.
        """
    },
    {
        "id": "tech_003",
        "title": "Vector Databases: FAISS vs ChromaDB",
        "category": "Technical",
        "content": """
        VECTOR DATABASES FOR RAG SYSTEMS

        Vector databases store high-dimensional embeddings and enable fast approximate
        nearest neighbor (ANN) search, which is the core retrieval operation in RAG.

        FAISS (Facebook AI Similarity Search)
        FAISS is a library for efficient similarity search developed by Meta AI.
        It supports multiple index types:
        - IndexFlatL2: Exact brute-force L2 distance (accurate, slow at scale)
        - IndexFlatIP: Exact inner product (cosine similarity with normalized vectors)
        - IndexIVFFlat: Inverted file index — clusters vectors, searches subset (fast)
        - IndexHNSW: Hierarchical Navigable Small World graph (best speed/recall tradeoff)
        FAISS is purely in-memory, has no persistence layer, and requires manual serialization.
        Best for: high-performance production, when you control infrastructure.

        CHROMADB
        ChromaDB is an open-source embedding database with a simple Python API.
        It handles persistence, metadata filtering, and collection management natively.
        ChromaDB supports multiple backends (DuckDB for local, ClickHouse for production).
        It integrates directly with LangChain and LlamaIndex.
        Best for: prototyping, Streamlit demos, smaller-scale production.

        CHOOSING BETWEEN THEM
        Use FAISS when: you need maximum performance, billion-scale vectors, custom index tuning
        Use ChromaDB when: you need quick setup, metadata filtering, built-in persistence
        Use Pinecone/Weaviate/Qdrant when: you need managed cloud hosting, hybrid search at scale

        SIMILARITY METRICS
        Cosine similarity: measures angle between vectors, scale-invariant (most common for text)
        Dot product: cosine × magnitude, faster computation, use with normalized embeddings
        L2 (Euclidean): measures absolute distance, sensitive to vector magnitude
        """
    },
    {
        "id": "tech_004",
        "title": "Embeddings & Semantic Search",
        "category": "Technical",
        "content": """
        TEXT EMBEDDINGS FOR SEMANTIC SEARCH

        Text embeddings are dense vector representations that capture semantic meaning.
        Unlike sparse bag-of-words representations, embeddings encode meaning such that
        similar concepts are close in vector space regardless of exact word match.

        EMBEDDING MODELS
        Word2Vec (2013): Predicts surrounding words, produces word-level embeddings
        GloVe (2014): Global co-occurrence statistics, word-level
        BERT (2018): Bidirectional Transformer, produces contextual token embeddings
        Sentence-BERT (2019): BERT fine-tuned with siamese network for sentence similarity
        OpenAI text-embedding-ada-002: 1536-dim embeddings, strong general performance
        BGE-M3: State-of-the-art open-source multi-lingual embedding model (2024)

        TF-IDF AS A BASELINE
        TF-IDF (Term Frequency-Inverse Document Frequency) is a classic sparse embedding.
        TF measures how often a term appears in a document.
        IDF penalizes terms common across all documents (like "the", "is").
        TF-IDF score = TF(t,d) × log(N / df(t))
        While less powerful than neural embeddings, TF-IDF is interpretable and fast.
        It forms the backbone of BM25, still competitive for keyword-heavy search.

        CHUNKING AFFECTS EMBEDDING QUALITY
        Embedding quality degrades if chunks are too long (semantic dilution) or too short
        (insufficient context). Optimal chunk sizes are typically 256-512 tokens for
        most embedding models. The embedding model's context window is the hard upper limit.

        EMBEDDING DIMENSIONS & TRADEOFFS
        Higher dimensions: more expressive, but slower search and more memory
        Lower dimensions: faster, cheaper, but may lose nuance
        Matryoshka embeddings (MRL): can be truncated at inference time without retraining
        """
    },
    {
        "id": "tech_005",
        "title": "LLM Hallucinations: Causes & Mitigations",
        "category": "Technical",
        "content": """
        LLM HALLUCINATIONS: CAUSES, TYPES & MITIGATIONS

        Hallucination refers to LLM outputs that are fluent and confident but factually
        incorrect, fabricated, or unsupported by any real source. It is one of the primary
        obstacles to deploying LLMs in production systems.

        TYPES OF HALLUCINATION
        Factual hallucination: The model states false facts with confidence
          Example: "The Eiffel Tower was built in 1756" (correct: 1889)
        Source hallucination: The model fabricates citations, papers, or URLs
          Example: Citing a non-existent paper "Smith et al. 2019"
        Instruction hallucination: The model ignores constraints in the prompt
          Example: Asked to summarize only, it adds unrequested analysis
        Consistency hallucination: The model contradicts itself within one response

        ROOT CAUSES
        1. Training data noise: LLMs learn from internet text which contains errors
        2. Knowledge cutoff: Models don't know events after their training date
        3. Overconfidence: RLHF tuning rewards confident, helpful responses
        4. Distributional shift: Questions outside training distribution trigger guessing
        5. Long context compression: Important facts get "forgotten" in long contexts

        MITIGATION STRATEGIES
        RAG (Retrieval-Augmented Generation): Ground answers in retrieved documents
        Chain-of-Thought prompting: Force step-by-step reasoning before answering
        Self-consistency: Sample multiple responses and take majority vote
        Constitutional AI: Model critiques its own outputs for factual claims
        Grounding score: Measure overlap between response and source documents
        Abstention: Train models to say "I don't know" when uncertain

        MEASURING HALLUCINATION
        ROUGE/BLEU: Overlap between response and source (rough proxy)
        BERTScore: Semantic similarity between response and retrieved context
        FActScoring: Decompose response into atomic claims, verify each against sources
        Groundedness ratio: Fraction of response sentences attributable to sources
        """
    },
    {
        "id": "tech_006",
        "title": "ML System Design: Production Best Practices",
        "category": "Technical",
        "content": """
        ML SYSTEM DESIGN FOR PRODUCTION

        FEATURE STORES
        A feature store centralizes computed features for training and serving.
        This prevents training-serving skew — where features computed differently
        at train time vs inference time cause silent model degradation.
        Popular options: Feast (open source), Tecton, Vertex AI Feature Store.

        MODEL VERSIONING & REGISTRY
        All production models must be versioned and stored in a model registry.
        Each version tracks: training data snapshot, hyperparameters, evaluation metrics,
        feature schema, and deployment history. MLflow and Weights & Biases are common tools.

        MONITORING & DRIFT DETECTION
        Data drift: Input feature distributions shift over time (e.g., customer behavior changes)
        Concept drift: The relationship between features and labels changes
        Model drift: Performance degrades without distribution shift (e.g., seasonality)
        Monitoring tools: Evidently AI, WhyLabs, Arize, custom statistical tests (KS, PSI).

        CI/CD FOR ML (MLOps)
        ML pipelines should be version controlled, tested, and automatically deployed.
        Key stages: data validation → training → evaluation → shadow deployment → canary → full rollout
        Shadow mode: New model runs in parallel, predictions logged but not served (safe testing)
        Canary deployment: Serve new model to 5% of traffic, monitor, then gradually increase

        LATENCY REQUIREMENTS
        Online inference: < 100ms p99 (user-facing features)
        Near-real-time: < 1s (fraud detection, recommendations)
        Batch: hours acceptable (churn scoring, lead scoring)
        LLM inference: 1-5s acceptable for most applications (streaming helps UX)
        """
    },
]

ALL_DOCUMENTS = HR_DOCS + TECH_DOCS

def get_documents():
    return ALL_DOCUMENTS

def get_documents_by_category(category: str):
    return [d for d in ALL_DOCUMENTS if d["category"] == category]

# Questions the RAG system should answer well (with sources)
# vs questions where vanilla LLM will hallucinate
DEMO_QUESTIONS = {
    "hr": [
        "How many days of annual leave do employees get?",
        "What is the home office allowance amount?",
        "How does the promotion process work?",
        "What happens if an employee gets a rating of 2?",
        "What is the 401K matching policy?",
        "How much internet reimbursement can remote workers claim?",
    ],
    "technical": [
        "What are the main components of a RAG pipeline?",
        "What is the difference between FAISS and ChromaDB?",
        "How does self-attention work in Transformers?",
        "What are the main causes of LLM hallucinations?",
        "What is TF-IDF and how does it work?",
        "What salary does a Senior engineer (L3) earn at TechCorp?",
    ],
    "hallucination_traps": [
        # These will expose vanilla LLM hallucination — company-specific info it can't know
        "What is TechCorp's exact parental leave policy?",
        "How much is the learning and development budget per employee?",
        "What are TechCorp's salary bands for 2024?",
        "What VPN policy does TechCorp enforce for remote workers?",
        "When are TechCorp's core collaboration days?",
    ]
}

if __name__ == "__main__":
    print(f"Total documents: {len(ALL_DOCUMENTS)}")
    for doc in ALL_DOCUMENTS:
        print(f"  [{doc['category']:10s}] {doc['id']}: {doc['title']}")
