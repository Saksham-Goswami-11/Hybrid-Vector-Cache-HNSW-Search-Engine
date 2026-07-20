export const MOCK_DATA = {
    "concurrent reads": {
        latency: 3.8, // Base latency
        results: [
            {
                score: 0.9823,
                id: "chunk:42",
                metadata: { namespace: "docs", source: "paper.pdf", page: 7 },
                text: "...uses a sync.RWMutex so multiple goroutines can hold read locks simultaneously..."
            },
            {
                score: 0.7441,
                id: "chunk:17",
                metadata: { namespace: "docs", source: "readme.md", page: 2 },
                text: "...the connection goroutine reads from its own bufio.Reader without touching..."
            },
            {
                score: 0.6102,
                id: "chunk:88",
                metadata: { namespace: "docs", source: "design.md", page: 14 },
                text: "...GOMAXPROCS workers in the pool means reads are distributed across cores..."
            }
        ]
    },
    "what happens when server restarts": {
        latency: 4.2,
        results: [
            {
                score: 0.9711,
                id: "chunk:31",
                metadata: { namespace: "docs", source: "architecture.md", page: 1 },
                text: "...the AOF log replays every VSET command in sequence on startup..."
            },
            {
                score: 0.8203,
                id: "chunk:55",
                metadata: { namespace: "docs", source: "backup.md", page: 3 },
                text: "...BGSAVE triggers an async snapshot — the server continues accepting connections..."
            },
            {
                score: 0.7018,
                id: "chunk:12",
                metadata: { namespace: "docs", source: "persistence.md", page: 5 },
                text: "...CRC32 checksum per AOF entry ensures corrupted lines are skipped with a warning..."
            }
        ]
    },
    "default": {
        latency: 4.6,
        results: [
            {
                score: 0.5421,
                id: "chunk:102",
                metadata: { namespace: "docs", source: "faq.md", page: 1 },
                text: "...No exact matches found. Synapse Cache uses brute-force exact cosine similarity..."
            },
            {
                score: 0.4211,
                id: "chunk:05",
                metadata: { namespace: "docs", source: "intro.md", page: 1 },
                text: "...Synapse Cache is an in-memory vector database written in Go..."
            }
        ]
    }
};
