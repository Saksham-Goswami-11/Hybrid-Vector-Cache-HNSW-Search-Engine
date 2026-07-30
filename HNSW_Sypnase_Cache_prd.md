__SYNAPSE CACHE__

__v2\.0 HNSW Index__

Hierarchical Navigable Small World Graph Implementation

Product Requirements Document · Go from Scratch · O\(log N\) ANN Search

__Document Type__

Product Requirements Document

__Version__

2\.0 HNSW Index Feature

__Status__

Ready for Implementation

__Depends On__

Synapse Cache v1\.0 \(brute\-force engine\)

__Algorithm__

Malkov & Yashunin, 2016 \(arXiv:1603\.09320\)

__Language__

Go zero external dependencies

__Last Updated__

June 2025

github\.com/your\-handle/synapse\-cache/tree/feat/hnsw

# __1\. Why HNSW The Scaling Problem__

Synapse Cache v1\.0 ships a brute\-force exact similarity engine\. Every VSIMILARITY query computes cosine similarity against every vector in the namespace\. This is the correct default it is simple, auditable, and exact\. The benchmark story for v1\.0 is built on this foundation\.

But brute\-force has a fundamental scaling limit\. It is O\(N\) linear in the number of stored vectors\. Here is what that means in practice:

__Corpus size__

__Brute\-force p50__

__HNSW p50 \(target\)__

__Speedup__

__10,000 vectors__

4\.5ms

0\.3ms

~15×

__100,000 vectors__

42ms

0\.8ms

~52×

__500,000 vectors__

210ms

1\.2ms

~175×

__1,000,000 vectors__

420ms

1\.5ms

~280×

__10,000,000 vectors__

4,200ms

2\.1ms

~2000×

__The Inflection Point__

Brute\-force is correct for corpora under ~500K vectors which covers the vast majority of single\-application RAG pipelines \(a 500\-page book is ~5K chunks at 512 tokens\)\. HNSW is the right choice once you are building multi\-tenant systems, enterprise knowledge bases, or anything that ingests documents continuously\. v2\.0 makes HNSW an opt\-in mode; brute\-force remains the default\.

HNSW achieves O\(log N\) approximate nearest neighbor search by organizing vectors into a multi\-layer navigable graph\. It trades a small amount of recall accuracy \(typically 9599%\) for logarithmic scaling\. This tradeoff is well\-understood and universally accepted by production vector databases \(Pinecone, Qdrant, Weaviate, Milvus all use HNSW or a variant\)\.

# __2\. Algorithm Deep Dive How HNSW Works__

This section is the most important part of the PRD\. You must understand the algorithm completely before writing a single line of code\. HNSW has no complex math it is a graph algorithm\. Its elegance is architectural\.

## __2\.1 The Foundation Navigable Small World \(NSW\)__

Before HNSW, there was NSW\. The insight: if you build a graph where each vector is connected to its M nearest neighbors, you can navigate from any entry point to the nearest neighbor of a query in sub\-linear time using a greedy search\.

Greedy search works like this: start at some node, look at its neighbors, move to whichever neighbor is closest to your query\. Repeat until no neighbor is closer than your current node\. That local minimum is your approximate nearest neighbor\.

__The NSW Problem__

NSW graphs built on insertion order create 'hubs' early\-inserted nodes accumulate enormous connection counts because they are the only nodes available when later nodes are inserted\. These hubs become bottlenecks\. Every search routes through them\. NSW degrades on clustered or high\-dimensional data\. HNSW solves this\.

## __2\.2 The HNSW Innovation Hierarchical Layers__

HNSW adds a probabilistic layer assignment\. When a new vector is inserted, its maximum layer is sampled from an exponentially decaying distribution:

// Layer assignment from the Malkov & Yashunin paper

func \(h \*HNSW\) randomLevel\(\) int \{

 // mL = 1 / ln\(M\) the level multiplier

 // Each level is exponentially less likely

 level := int\(\-math\.Log\(rand\.Float64\(\)\) \* h\.mL\)

 if level > h\.maxLevel \{

 level = h\.maxLevel

 \}

 return level

\}

 

// With M=16, mL ≈ 0\.361

// P\(level=0\) ≈ 69\.8%

// P\(level=1\) ≈ 21\.3%

// P\(level=2\) ≈ 6\.5%

// P\(level=3\) ≈ 2\.0%

// \.\.\. exponentially decreasing

This creates a skip\-list\-like structure in vector space:

- Layer 0 \(ground layer\): contains ALL vectors\. Each node has up to 2×M connections\. This is where the final fine\-grained search happens\.
- Layer 1: contains ~1/e ≈ 37% of vectors\. Connections span larger distances long\-range navigation\.
- Layer 2: contains ~1/e² ≈ 14% of vectors\. Even longer range\.
- Layer L \(top\): contains just a few vectors\. The entry point into the entire graph\.

## __2\.3 The Insert Algorithm \(Algorithm 1 in the paper\)__

This is the most important algorithm to implement correctly\. Every future query depends on the quality of insertions\.

INSERT\(hnsw, q, M, Mmax, efConstruction, mL\):

 

 W = \[\] // list of currently found nearest elements

 ep = hnsw\.entryPoint

 L = hnsw\.currentMaxLayer

 l = randomLevel\(mL\) // assigned layer for q

 

 // Phase 1: Greedy descent from top to l\+1

 // \(just find the entry point for the insertion layer\)

 for lc = L down to l\+1:

 W = SEARCH\_LAYER\(q, ep, ef=1, lc\)

 ep = nearest element in W to q

 

 // Phase 2: At each layer from l down to 0,

 // find neighbors with efConstruction candidates

 for lc = min\(L, l\) down to 0:

 W = SEARCH\_LAYER\(q, ep, ef=efConstruction, lc\)

 neighbors = SELECT\_NEIGHBORS\(q, W, M, lc\) // pick best M

 add bidirectional connections between q and neighbors at lc

 

 // Prune existing nodes that now exceed Mmax connections

 for each e in neighbors:

 eConn = neighborhood\(e, lc\)

 if len\(eConn\) > Mmax:

 eNewConn = SELECT\_NEIGHBORS\(e, eConn, Mmax, lc\)

 set neighborhood\(e, lc\) = eNewConn

 

 ep = W // use W as entry points for next layer

 

 // Update entry point if q reaches a new top layer

 if l > L:

 hnsw\.entryPoint = q

 hnsw\.currentMaxLayer = l

## __2\.4 The Search Algorithm \(Algorithm 2 in the paper\)__

KNN\_SEARCH\(hnsw, q, K, ef\):

 

 W = \[\] // dynamic candidate list

 ep = hnsw\.entryPoint

 L = hnsw\.currentMaxLayer

 

 // Phase 1: Greedy descent from top to layer 1

 // ef=1 means we just track the single best entry point

 for lc = L down to 1:

 W = SEARCH\_LAYER\(q, ep, ef=1, lc\)

 ep = nearest element in W to q

 

 // Phase 2: Full search at ground layer with ef candidates

 W = SEARCH\_LAYER\(q, ep, ef=max\(ef, K\), layer=0\)

 

 return K nearest elements from W

## __2\.5 SEARCH\_LAYER The Core Subroutine__

This is called by both insert and search\. It performs a greedy best\-first traversal of a single graph layer using two priority queues:

SEARCH\_LAYER\(q, ep, ef, lc\):

 

 visited = set\(ep\)

 candidates = min\-heap by distance to q // to explore next

 found = max\-heap by distance to q // best ef found so far

 

 add ep to both candidates and found

 

 while candidates is not empty:

 c = extract\-min from candidates // closest unvisited

 f = furthest in found // worst of our best ef

 

 if distance\(c, q\) > distance\(f, q\):

 break // all remaining candidates are worse than our worst found

 

 for each neighbor e of c at layer lc:

 if e not in visited:

 visited\.add\(e\)

 f = furthest in found

 if distance\(e, q\) < distance\(f, q\) or len\(found\) < ef:

 add e to candidates

 add e to found

 if len\(found\) > ef:

 remove furthest from found

 

 return found // ef nearest elements

## __2\.6 SELECT\_NEIGHBORS The Heuristic__

The paper presents two neighbor selection strategies\. Use the heuristic version \(Algorithm 4\) it produces significantly better recall on clustered and high\-dimensional data, which is exactly what text embeddings are\.

Simple strategy: just take the M closest candidates\. Fast, but causes clustering artifacts\.

Heuristic strategy: when evaluating candidate e, only add it if it is closer to the query than to any already\-selected neighbor\. This ensures diversity neighbors 'spread out' in vector space rather than all clustering near the query\.

SELECT\_NEIGHBORS\_HEURISTIC\(q, C, M, lc, extendCandidates, keepPrunedConnections\):

 

 R = \[\] // result

 W = C // working queue \(copy of candidates\)

 

 if extendCandidates:

 for each e in C:

 for each eAdj in neighborhood\(e, lc\):

 if eAdj not in W: add eAdj to W

 

 Wd = \[\] // discarded candidates

 

 while len\(W\) > 0 and len\(R\) < M:

 e = extract\-nearest\-to\-q from W

 

 if e is closer to q than to any element in R:

 add e to R // e is a 'useful' diverse neighbor

 else:

 add e to Wd // e is redundant

 

 if keepPrunedConnections:

 while len\(Wd\) > 0 and len\(R\) < M:

 add extract\-nearest\-to\-q\(Wd\) to R

 

 return R

# __3\. HNSW Parameters Complete Guide__

HNSW has three tuning parameters\. Understanding them deeply is what separates someone who used a library from someone who built one\. These will be asked in every interview\.

## __3\.1 M Maximum Connections per Node__

__What M actually controls__

M is the number of bidirectional connections each node maintains in layers 1 and above\. Layer 0 uses 2×M connections\. M is the single most important parameter for recall quality and memory usage\. It is set at index construction time and cannot be changed without rebuilding the entire graph\.

__M value__

__Memory impact__

__Recall impact__

__Use case__

__48__

Minimal

Lower \(~8590%\)

Memory\-constrained, approximate search acceptable

__1216__

Moderate

Good \(~9296%\)

Default\. Recommended for most RAG workloads

__2432__

2× default

High \(~9699%\)

Production RAG, high\-stakes retrieval

__4864__

4× default

Very high \(~99%\+\)

Research, max\-recall applications

__64\+__

Diminishing

Marginal gains

Almost never justified

Memory formula: each vector in layer 0 stores up to 2M node IDs \(uint32\) plus the float32 vector data\. At M=16, dim=1536:

// Memory per vector at M=16, dim=1536

vectorData = 1536 \* 4 bytes = 6,144 bytes

layer0Links = 2\*16 \* 4 bytes \(uint32\) = 128 bytes

upperLayerLinks≈ 16 \* 4 bytes \(average\) = 64 bytes \(most nodes in layer 0 only\)

metadata = ~100 bytes \(estimate\)

 ≈ 6,436 bytes per vector

 

// 100K vectors at M=16:

100,000 \* 6,436 = ~644 MB

 

// Compare to brute\-force at same corpus:

100,000 \* \(1536\*4\) = ~614 MB \(only slightly less the graph overhead is small\)

## __3\.2 efConstruction Build\-Time Search Depth__

efConstruction controls how many candidates are considered when finding neighbors for each newly inserted vector during graph construction\. Higher values build a better\-quality graph but take longer\. It has zero effect on index memory or query speed only on build quality\.

__efConstruction__

__Build time impact__

__Recall impact__

__Recommendation__

__4080__

Fastest

Acceptable for testing

Development / CI only

__100200__

23× faster build

Good production default

Default: 200

__200400__

Moderate

High recall

High\-stakes deployments

__400\+__

Slow

Diminishing returns

Not recommended

__The Rule of Thumb__

Set efConstruction ≥ M\. The paper's recommendation and all production databases agree: efConstruction=200 with M=16 is the sweet spot for the vast majority of workloads\. Going above 200 rarely improves recall more than 12%, but doubles or triples build time\.

## __3\.3 ef \(efSearch\) Query\-Time Search Depth__

ef controls the size of the candidate list during search\. Unlike M and efConstruction, ef can be changed per\-query at runtime you do not need to rebuild the index\. This is the primary knob for the recall\-latency tradeoff at query time\.

__ef value__

__Latency impact__

__Recall impact__

__Use case__

__K \(minimum\)__

Fastest

~8090%

Non\-critical, speed\-first

__50__

Fast

~9295%

Good default for most queries

__100__

Moderate

~9698%

Production RAG default

__200__

Slower

~9899%

High\-stakes retrieval

__500\+__

Approaches brute\-force

99%\+

Near\-exact, large K searches

Rule: ef must always be ≥ K \(the number of results requested\)\. ef = max\(ef, K\) is enforced in the implementation\.

# __4\. Go Data Structures__

## __4\.1 Node Representation__

// internal/hnsw/node\.go

 

type Node struct \{

 ID uint32 // internal integer ID \(maps to string key\)

 Vector \[\]float32 // the embedding stored by reference

 Metadata map\[string\]string // arbitrary key\-value pairs

 Links \[\]\[\]uint32 // Links\[layer\] = \[\]neighborID

 // Links\[0\] has up to 2\*M neighbors \(ground layer\)

 // Links\[1\.\.L\] have up to M neighbors each

 mu sync\.RWMutex // per\-node lock for link mutation

\}

 

// Why per\-node locks?

// During concurrent inserts, different goroutines modify different

// nodes' link lists\. A global lock would serialize all inserts\.

// Per\-node locks allow N concurrent inserts to proceed in parallel

// as long as they modify different nodes \(which is almost always\)\.

## __4\.2 Index Structure__

// internal/hnsw/index\.go

 

type Index struct \{

 // Configuration set at construction, immutable

 M int // max connections per node \(default 16\)

 Mmax int // max connections at layer 0 = 2\*M

 Mmax0 int // alias: Mmax = 2\*M

 efConstruction int // build\-time candidate list size \(default 200\)

 ef int // query\-time candidate list size \(default 100\)

 mL float64 // level multiplier = 1/ln\(M\)

 maxLevel int // safety cap on layer assignment \(default 16\)

 

 // Graph state protected by mu

 nodes \[\]\*Node // all nodes, indexed by internal ID

 idMap map\[string\]uint32 // string key → internal ID

 reverseMap \[\]string // internal ID → string key

 entryPoint uint32 // ID of the entry point node

 currentMaxLayer int // highest layer currently in use

 mu sync\.RWMutex

 

 // Stats

 insertCount atomic\.Int64

 searchCount atomic\.Int64

\}

 

func NewIndex\(M, efConstruction int\) \*Index \{

 return &Index\{

 M: M,

 Mmax: M,

 Mmax0: 2 \* M,

 efConstruction: efConstruction,

 ef: 100, // default efSearch

 mL: 1\.0 / math\.Log\(float64\(M\)\),

 maxLevel: 16,

 idMap: make\(map\[string\]uint32\),

 \}

\}

## __4\.3 Priority Queue Implementation__

SEARCH\_LAYER requires two heaps: a min\-heap for candidates \(so we always explore the closest next\) and a max\-heap for results \(so we can evict the furthest when we exceed ef\)\. Go's container/heap with a custom Less function handles both just negate the comparator for the max\-heap\.

// internal/hnsw/heap\.go

 

type Item struct \{

 id uint32

 dist float32

\}

 

// MinHeap pop returns the item with SMALLEST distance \(for candidates\)

type MinHeap \[\]Item

func \(h MinHeap\) Less\(i, j int\) bool \{ return h\[i\]\.dist < h\[j\]\.dist \}

// \.\.\. Len, Swap, Push, Pop standard container/heap interface

 

// MaxHeap pop returns the item with LARGEST distance \(for found set\)

type MaxHeap \[\]Item

func \(h MaxHeap\) Less\(i, j int\) bool \{ return h\[i\]\.dist > h\[j\]\.dist \}

// \.\.\. same interface

 

// Why both?

// candidates \(MinHeap\): we always want to explore the closest unvisited node first

// found \(MaxHeap\): we want to quickly evict the furthest node when len > ef

# __5\. Concurrency Model__

Concurrent HNSW is the hardest engineering problem in this project\. The original paper describes a single\-threaded algorithm\. Making it concurrent without data races requires careful thought\.

## __5\.1 The Problem__

During insertion, the algorithm modifies the link lists of multiple nodes\. Two concurrent insertions can both try to modify the same node's link list at the same time\. A global write lock would work but would serialize all inserts defeating the purpose of goroutines\.

## __5\.2 The Solution Per\-Node Locking__

__The Locking Strategy__

Each Node has its own sync\.RWMutex\. Reads \(neighbor traversal during SEARCH\_LAYER\) take a RLock on each visited node's link list multiple concurrent searches traverse the graph simultaneously\. Writes \(adding a bidirectional link during INSERT\) take a full Lock on the specific node being modified\. Since inserts modify small, specific nodes rather than the whole graph, contention is minimal in practice\.

// Correct bidirectional link insertion with per\-node locking

func \(idx \*Index\) addLink\(fromID, toID uint32, layer int\) \{

 // Lock BOTH nodes in consistent order \(lower ID first\)

 // to prevent deadlock

 a, b := fromID, toID

 if a > b \{ a, b = b, a \}

 

 idx\.nodes\[a\]\.mu\.Lock\(\)

 idx\.nodes\[b\]\.mu\.Lock\(\)

 defer idx\.nodes\[a\]\.mu\.Unlock\(\)

 defer idx\.nodes\[b\]\.mu\.Unlock\(\)

 

 // Add bidirectional links

 idx\.nodes\[fromID\]\.Links\[layer\] = append\(idx\.nodes\[fromID\]\.Links\[layer\], toID\)

 idx\.nodes\[toID\]\.Links\[layer\] = append\(idx\.nodes\[toID\]\.Links\[layer\], fromID\)

\}

 

// CRITICAL: Always lock lower ID first\.

// Goroutine A holds lock\(5\), wants lock\(8\)

// Goroutine B holds lock\(8\), wants lock\(5\)

// Without ordering: deadlock\.

// With ordering \(both lock 5 then 8\): one waits, no deadlock\.

## __5\.3 Entry Point and Global State__

The index's entry point and currentMaxLayer are shared global state\. These are protected by the index\-level sync\.RWMutex\. Reads of the entry point \(at the start of every search and insert\) take idx\.mu\.RLock\(\)\. Updates to the entry point \(when a new node reaches a higher layer\) take idx\.mu\.Lock\(\)\.

This means entry point updates are rare \(only when a newly inserted node gets assigned a higher layer than currently exists probability decreases exponentially\) and thus the global lock is rarely contended\.

## __5\.4 The visited Set in SEARCH\_LAYER__

The visited set must be per\-goroutine not shared\. Each concurrent search has its own visited set\. This is naturally handled by Go's stack allocation since SEARCH\_LAYER is called by independent goroutines, but it must be explicitly ensured: the visited set is always a local variable, never a field on the Index\.

# __6\. New Wire Protocol Commands__

HNSW is exposed as an opt\-in index mode per namespace\. The brute\-force engine remains the default\. A namespace that has had VINDEX CREATE run on it uses HNSW for all subsequent VSIMILARITY queries\.

## __6\.1 VINDEX Index Management__

### __VINDEX CREATE Build an HNSW index on an existing namespace__

VINDEX CREATE <namespace> \[M <m>\] \[EFC <efConstruction>\]

 

Response: \+OK \(synchronous blocks until index is fully built\)

 \-ERR namespace not found

 \-ERR namespace already has an index \(use VINDEX DROP first\)

 

Defaults: M=16, EFC=200

 

Examples:

 VINDEX CREATE docs

 VINDEX CREATE docs M 32 EFC 400

 

// After this command, all VSIMILARITY queries on 'docs'

// automatically use HNSW\. No client\-side changes needed\.

### __VINDEX DROP Remove the HNSW index, revert to brute\-force__

VINDEX DROP <namespace>

 

Response: \+OK

 \-ERR no index exists for namespace

 

// Vectors are preserved\. Only the graph is discarded\.

// VSIMILARITY reverts to O\(N\) brute\-force scan\.

### __VINDEX INFO Inspect the index__

VINDEX INFO <namespace>

 

Response: bulk string with fields:

 index\_type:hnsw

 M:16

 ef\_construction:200

 ef\_search:100

 max\_layer:4

 entry\_point\_id:chunk:8821

 total\_nodes:48291

 total\_links:1544352

 memory\_bytes:316841984

 build\_time\_ms:4821

 index\_recall\_estimate:0\.97 // measured against brute\-force sample

### __VINDEX SET\_EF Adjust efSearch at runtime \(no rebuild\)__

VINDEX SET\_EF <namespace> <ef>

 

Response: \+OK

 \-ERR ef must be >= 1

 

// ef controls the recall\-latency tradeoff at query time\.

// Higher ef = better recall, higher latency\.

// Default: 100

 

// Example: switch to high\-recall mode for an important query

VINDEX SET\_EF docs 300

VSIMILARITY docs 4 0\.18 \-0\.44 0\.99 0\.00 TOP 5

VINDEX SET\_EF docs 100 // reset

## __6\.2 VSET Change \(Incremental Insert\)__

When an HNSW index exists on a namespace, VSET automatically inserts the new vector into the graph\. No rebuild required\. This is one of HNSW's key advantages over IVF\-style indexes it supports online insertion natively\.

// Client sends:

VSET docs new\_chunk:9999 4 0\.21 \-0\.31 0\.88 0\.12 META source new\_paper\.pdf

 

// Server behavior \(pseudocode\):

if namespace has HNSW index:

 store vector in namespace map \(as before\)

 hnsw\.Insert\(new\_chunk:9999, vector, metadata\) // O\(log N\)

else:

 store vector in namespace map \(as before\)

 // brute\-force scan will pick it up automatically

 

// Response: \+OK \(same as before client sees no difference\)

## __6\.3 VSIMILARITY Transparent Routing__

The VSIMILARITY command interface does not change\. The server internally routes to HNSW or brute\-force based on whether the namespace has an active index\. The response format is identical\.

// Same command, same response format

VSIMILARITY docs 4 0\.18 \-0\.44 0\.99 0\.00 TOP 5

 

// Server routing logic:

if hnsw\.HasIndex\(namespace\):

 results = hnsw\.Search\(queryVector, K=5, ef=index\.ef\)

else:

 results = bruteForce\.TopK\(namespace, queryVector, K=5\)

 

// This is a key design decision: the client never needs to know

// which engine is running\. You can switch between them with

// VINDEX CREATE / VINDEX DROP without touching client code\.

# __7\. Implementation Plan__

__Before You Write Code__

Read the original paper \(arXiv:1603\.09320\) specifically Algorithms 1, 2, 3, and 4\. It is 12 pages\. Read it twice\. Then read the visual guide at cfu288\.com/blog/2024\-05\_visual\-guide\-to\-hnsw \(it has interactive animations\)\. Only after you can explain all four algorithms from memory should you open your editor\.

__Phase__

__Duration__

__Deliverables__

__P1__

Week 1

Read paper \+ visual guides\. Draw the data structures on paper\. Write types\.go and heap\.go with full tests\. Do not touch Insert or Search yet\.

__P2__

Week 2

Implement SEARCH\_LAYER \(single\-threaded\)\. Write unit tests: verify it finds the exact nearest neighbor in a small hand\-built graph\. This is your correctness baseline\.

__P3__

Week 3

Implement INSERT \(single\-threaded\)\. Build a small index of 1,000 random vectors\. Run SEARCH\_LAYER against it\. Measure recall@10 vs brute\-force\. Target: >85%\.

__P4__

Week 4

Implement SELECT\_NEIGHBORS\_HEURISTIC\. Replace simple strategy\. Re\-run recall tests\. Target: >92%\. Debug until you hit it\.

__P5__

Week 5

Add concurrency: per\-node locks, consistent lock ordering, concurrent Insert tests under go \-race\. Zero races must pass before moving on\.

__P6__

Week 6

Wire into Synapse Cache store\. Add VINDEX CREATE/DROP/INFO/SET\_EF commands\. VSIMILARITY transparent routing\. Integration tests\.

__P7__

Week 7

AOF persistence for HNSW: serialize the graph to disk on BGSAVE, rebuild on startup\. Snapshot format design\.

__P8__

Week 8

Benchmarks, README updates, recall measurement tooling, PR merge\.

## __7\.1 Phase 1 Data Structures Only \(Week 1\)__

Do not touch Insert or Search in week 1\. Only write:

1. internal/hnsw/node\.go Node struct, NewNode, AddLink, GetLinks
2. internal/hnsw/index\.go Index struct, NewIndex, randomLevel
3. internal/hnsw/heap\.go MinHeap and MaxHeap with full container/heap interface
4. internal/hnsw/heap\_test\.go exhaustive tests: push 1000 items, verify pop order

Why? Because getting the heap wrong means every single search produces wrong results and the bug is invisible\. Verify your heaps are perfect before building anything on top of them\.

## __7\.2 Phase 2 SEARCH\_LAYER \(Week 2\)__

Build a tiny hand\-crafted graph of 5 nodes in 2D space\. Run SEARCH\_LAYER\. Verify by hand that it finds the right answer\. Then test at 100 nodes with random 2D vectors compare to brute\-force\. They should agree 100% of the time on a well\-connected 2D graph\.

func TestSearchLayer\_TinyGraph\(t \*testing\.T\) \{

 // 5 vectors in 2D, hand\-built graph

 // \[1,0\], \[0\.9,0\.1\], \[0,1\], \[\-1,0\], \[0,\-1\]

 // Entry point: \[1,0\]

 // Query: \[0\.95, 0\.05\] should find \[0\.9,0\.1\] as nearest

 

 idx := buildTinyTestGraph\(\)

 results := idx\.searchLayer\(query, entryPointID, ef=3, layer=0\)

 assert nearest result is nodeID of \[0\.9, 0\.1\]

\}

## __7\.3 Phase 34 Insert and Recall Testing \(Weeks 34\)__

The recall measurement framework is not optional it is how you know your implementation is correct\. Build it before you try to tune:

// bench/recall\_test\.go

 

func measureRecall\(t \*testing\.T, idx \*hnsw\.Index, queries \[\]\[\]float32,

 groundTruth \[\]\[\]uint32, K int\) float64 \{

 // groundTruth\[i\] = the K true nearest neighbors of queries\[i\]

 // \(computed by brute\-force before the test\)

 

 var totalFound int

 for i, q := range queries \{

 results := idx\.Search\(q, K, idx\.EF\)

 found := intersect\(results, groundTruth\[i\]\)

 totalFound \+= len\(found\)

 \}

 return float64\(totalFound\) / float64\(len\(queries\)\*K\)

\}

 

// Target recalls:

// 1,000 vectors, dim=128: >92% at ef=100

// 10,000 vectors, dim=1536: >90% at ef=100, >96% at ef=300

## __7\.4 Phase 5 Concurrency \(Week 5\)__

The race detector test is your gate\. You must not move to Phase 6 until this passes cleanly:

func TestHNSW\_ConcurrentInsert\(t \*testing\.T\) \{

 idx := hnsw\.NewIndex\(16, 200\)

 var wg sync\.WaitGroup

 

 // 1000 goroutines, each inserting 10 vectors simultaneously

 for g := 0; g < 1000; g\+\+ \{

 wg\.Add\(1\)

 go func\(offset int\) \{

 defer wg\.Done\(\)

 for i := 0; i < 10; i\+\+ \{

 vec := randomVector\(1536\)

 idx\.Insert\(fmt\.Sprintf\("vec:%d", offset\*10\+i\), vec, nil\)

 \}

 \}\(g\)

 \}

 wg\.Wait\(\)

 

 // Verify: all 10,000 vectors are findable

 assert idx\.Len\(\) == 10000

\}

 

// Run with: go test \-race \-run TestHNSW\_ConcurrentInsert \./internal/hnsw/\.\.\.

# __8\. HNSW Persistence__

HNSW graphs are expensive to build a 100K\-vector index at M=16, efConstruction=200 takes several seconds\. Replaying the AOF log on startup \(as v1\.0 does for brute\-force\) would rebuild the graph from scratch on every restart\. This is unacceptable\.

Instead, HNSW snapshots are saved as a separate binary file alongside the AOF log\.

## __8\.1 Snapshot Format__

// synapse\.hnsw\.snap binary format

 

Header \(fixed 64 bytes\):

 \[8\] magic: 0x53594E4150534548 // 'SYNAPSEH'

 \[4\] version: 1

 \[4\] M: 16

 \[4\] efC: 200

 \[4\] dim: 1536 // must match stored vectors

 \[4\] nodeCount: N

 \[4\] maxLayer: L

 \[4\] entryID: E

 \[28\] reserved: zeros

 

For each node \(N nodes\):

 \[4\] internalID

 \[4\] keyLen

 \[keyLen\] key \(string, no null terminator\)

 \[4\] numLayers

 For each layer:

 \[4\] numLinks

 \[4\*numLinks\] linkIDs \(uint32\)

 // Note: vector data is NOT stored here\.

 // Vectors are loaded from the AOF log first,

 // then the graph is overlaid from this snapshot\.

## __8\.2 Startup Sequence__

1. Load vectors from AOF log into namespace maps \(existing v1\.0 behavior\)
2. For each namespace that has a \.hnsw\.snap file:
3. Validate snapshot header \(magic, version, dim match\)
4. Load graph structure node IDs, layer assignments, link lists
5. Cross\-reference: every snapshot node ID must exist in the loaded vector map
6. Set entry point and max layer from snapshot header
7. Namespace is now live VSIMILARITY uses HNSW

If the snapshot is missing or corrupt, the namespace falls back to brute\-force scan with a logged warning\. The index can be rebuilt with VINDEX CREATE\.

# __9\. Testing Strategy__

## __9\.1 Correctness Tests__

__Test__

__What it verifies__

__TestHeapMinPop__

MinHeap always returns smallest distance item

__TestHeapMaxPop__

MaxHeap always returns largest distance item

__TestRandomLevel__

randomLevel\(\) distribution matches expected exponential decay

__TestSearchLayerTiny__

SEARCH\_LAYER on 5\-node 2D graph finds correct nearest neighbor

__TestInsertSingle__

Insert 1 vector, search returns it with score=1\.0

__TestInsertDuplicate__

VSET same key twice second insert replaces first, no ghost nodes

__TestRecall\_1K\_dim128__

recall@10 > 90% on 1K vectors, dim=128, ef=100

__TestRecall\_10K\_dim1536__

recall@10 > 88% on 10K vectors, dim=1536, ef=100

__TestRecall\_ef300__

recall@10 > 95% on 10K vectors, dim=1536, ef=300

__TestSelectNeighborsHeuristic__

Heuristic recall > simple strategy recall on clustered data

__TestConcurrentInsert\_Race__

1000 concurrent inserts, go \-race: zero races

__TestConcurrentSearchInsert__

500 inserts and 500 searches simultaneously: no races, no panics

__TestSnapshotRoundTrip__

BGSAVE \+ restart: index state identical before and after

__TestCorruptSnapshot__

Truncated snapshot falls back to brute\-force, no panic, logged warning

## __9\.2 Benchmark Suite__

__Benchmark__

__Corpus__

__Target p50__

__Notes__

__BenchmarkHNSWInsert__

10K, dim\-1536

< 2ms/op

M=16, efC=200

__BenchmarkHNSWSearch__

10K, dim\-1536

< 0\.5ms

ef=100, K=10

__BenchmarkHNSWSearch__

100K, dim\-1536

< 1ms

ef=100, K=10

__BenchmarkHNSWSearch__

1M, dim\-1536

< 2ms

ef=100, K=10

__BenchmarkHNSWBuild__

100K, dim\-1536

< 30s total

M=16, efC=200, 1 goroutine

__BenchmarkHNSWBuildParallel__

100K, dim\-1536

< 8s total

M=16, efC=200, GOMAXPROCS

__BenchmarkRecall\_efSweep__

10K, dim\-1536

curve plot

ef=10,20,50,100,200,500

# __10\. README Additions The v2\.0 Story__

The README gets a new section after the benchmark table\. The narrative shift: v1\.0 proved Synapse Cache is faster than Redis on the similarity path\. v2\.0 proves it scales past the point where any Redis\-based solution collapses\.

## __10\.1 New Benchmark Table \(replaces v1\.0 table\)__

| Corpus | Brute\-Force | HNSW \(M=16, ef=100\) | Speedup |

|\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-|\-\-\-\-\-\-\-\-\-\-\-\-\-|\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-|\-\-\-\-\-\-\-\-\-|

| 1K × dim\-1536 | 0\.75ms | 0\.12ms | ~6× |

| 10K × dim\-1536 | 4\.5ms | 0\.42ms | ~10× |

| 100K × dim\-1536| 42ms | 0\.81ms | ~52× |

| 1M × dim\-1536 | 420ms | 1\.48ms | ~283× |

 

Recall@10 at ef=100: 93\.2% \(10K corpus, dim\-1536\)

Recall@10 at ef=300: 97\.8% \(10K corpus, dim\-1536\)

## __10\.2 New Quick Start Commands__

\# Start server

\./synapse \-\-port 6379

 

\# Load 10K vectors \(using the Go client\)

// \.\.\. VSET loop \.\.\.

 

\# Build HNSW index

echo \-e "VINDEX CREATE docs M 16 EFC 200\\r" | nc localhost 6379

\# \+OK \(takes ~2 seconds for 10K vectors\)

 

\# Query now uses HNSW automatically

echo \-e "VSIMILARITY docs 4 0\.18 \-0\.44 0\.99 0\.00 TOP 5\\r" | nc localhost 6379

 

\# Check index stats

echo \-e "VINDEX INFO docs\\r" | nc localhost 6379

## __10\.3 The Interview Story__

__What to say when asked about HNSW in an interview__

"I implemented HNSW from scratch in Go, directly from the Malkov & Yashunin paper\. The core insight is that you assign each vector to a random layer using an exponentially decaying probability like a skip list in vector space\. Searches start at the top layer navigating coarsely, then zoom in at lower layers\. The tricky engineering was concurrent insertion: I used per\-node RWMutex locks with consistent lock ordering to prevent deadlocks\. I verified correctness by measuring recall@10 against brute\-force ground truth 93% at ef=100, 97% at ef=300\. At 1M vectors, it returns results in 1\.5ms\. Brute\-force takes 420ms\."

# __11\. Risks & Mitigations__

__Risk__

__Severity__

__Mitigation__

__Incorrect SEARCH\_LAYER terminates too early or too late__

Critical

Unit test against hand\-built 2D graph with known answers\. Recall@10 test must pass before Phase 4\.

__Lock ordering deadlock in concurrent inserts__

Critical

Always lock lower nodeID first\. Write a dedicated deadlock test: 10K concurrent inserts with timeout assertion\.

__Recall degrades on real text embeddings__

High

OpenAI ada embeddings cluster differently than random vectors\. Test recall with real embeddings from your RAG demo corpus\.

__Graph 'orphan' nodes after high deletion rate__

Medium

v2\.0 has no VDEL\-from\-HNSW support \(see Non\-Goals\)\. Document this clearly\. v2\.1 adds soft\-delete with rebuild\.

__Snapshot corruption loses hours of build time__

Medium

CRC32 checksum in snapshot footer\. Validate on load\. Fall back to brute\-force \+ log warning\.

__Memory blowup at M=32\+__

Medium

Document memory formula clearly\. Add VINDEX INFO reporting memory\_bytes\. Test with M=16 as default; M=32 as opt\-in\.

__Project scope creep delays ship__

High

HNSW lives entirely on branch feat/hnsw\. Merge only when recall tests pass and race detector is clean\.

## __11\.1 Non\-Goals for v2\.0__

- __VDEL from HNSW graph__ Deleting a vector from an HNSW graph requires marking it as deleted and rebuilding affected links\. This is a v2\.1 feature\. In v2\.0, VDEL removes the vector from the namespace map but leaves it as an 'orphan' in the graph it wastes memory but does not corrupt results\.
- __Product quantization \(PQ\)__ Reduces memory by compressing vectors\. Out of scope\.
- __Multi\-vector queries__ Querying with multiple vectors and aggregating results\. Out of scope\.
- __Filtered search__ Pre\-filtering by metadata before similarity search\. Out of scope\.

# __12\. References & Required Reading__

These are not optional\. Read them in order before writing code\.

1. Malkov & Yashunin \(2016\)\. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs\. arXiv:1603\.09320\. The primary source\. Read Algorithms 14 carefully\.
2. Visual Guide to HNSW cfu288\.com/blog/2024\-05\_visual\-guide\-to\-hnsw Interactive animations of insert and search\. Watch the search animation until you can predict where it will go next\.
3. Pinecone HNSW Guide pinecone\.io/learn/series/faiss/hnsw Best intuitive explanation of the layer structure and parameter effects\.
4. OpenSearch HNSW Tuning opensearch\.org/blog/a\-practical\-guide\-to\-selecting\-hnsw\-hyperparameters Production M/efC/efSearch tuning guidance\.
5. Go container/heap docs pkg\.go\.dev/container/heap Know the interface cold\. Your MinHeap and MaxHeap implementations must be correct before anything else works\.
6. Go sync\.RWMutex docs pkg\.go\.dev/sync\#RWMutex Understand the writer\-starvation properties and when RLock vs Lock is appropriate\.

__Final Note__

When this is done, you will have implemented, from scratch in Go: a multi\-layer navigable graph, concurrent insert with per\-node locking and deadlock\-free lock ordering, a recall measurement framework, a binary snapshot format with CRC32 validation, and a new wire protocol extension\. That is not a side project\. That is a systems engineering portfolio piece that puts you in the same conversation as the engineers who built Qdrant and Weaviate\. Ship v1\.0 first\. Then build this\.

