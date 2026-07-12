package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/pkg/client"
)

type SampleItem struct {
	ID        string            `json:"id"`
	Text      string            `json:"text"`
	Embedding []float32         `json:"embedding"`
	Metadata  map[string]string `json:"metadata"`
}

type OpenAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func main() {
	addr := flag.String("addr", "localhost:6380", "Synapse Cache address")
	password := flag.String("password", "YourSuperSecurePassword123!", "Synapse Cache password")
	openAIKey := flag.String("openai-key", "", "OpenAI API Key (optional)")
	sampleFile := flag.String("sample-file", "data/sample_embeddings.json", "Path to sample embeddings JSON")
	flag.Parse()

	fmt.Println("==================================================")
	fmt.Println("Synapse Cache RAG Demo")
	fmt.Println("==================================================")

	c, err := client.New(client.Options{
		Addr:        *addr,
		Password:    *password,
		MaxConns:    5,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatalf("failed to connect to Synapse Cache: %v", err)
	}
	defer c.Close()
	fmt.Printf("connected to Synapse Cache at %s\n", *addr)

	ctx := context.Background()

	// 1. Ingest Data
	fmt.Println("loading and indexing documents...")
	err = loadData(ctx, c, *sampleFile)
	if err != nil {
		log.Fatalf("failed to load data: %v", err)
	}
	
	count, _ := c.VCount(ctx, "docs")
	fmt.Printf("indexed %d chunks into the 'docs' namespace.\n", count)
	fmt.Println("==================================================")

	// 2. REPL Loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nquery (or type 'quit'): ")
		query, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		query = strings.TrimSpace(query)
		if query == "quit" || query == "exit" {
			break
		}
		if query == "" {
			continue
		}

		fmt.Println("\nsearching...")
		var queryVec []float32

		if *openAIKey != "" {
			queryVec, err = getOpenAIEmbedding(query, *openAIKey)
			if err != nil {
				fmt.Printf("failed to get embedding: %v\n", err)
				continue
			}
		} else {
			fmt.Println("warning: no OpenAI key provided; generating a random query vector.")
			fmt.Println("(results will demonstrate speed, but semantic matches will be random)")
			queryVec = generateRandomVector(1536)
		}

		start := time.Now()
		results, err := c.VSimilarity(ctx, client.VSimilarityArgs{
			Namespace: "docs",
			Vector:    queryVec,
			TopK:      3,
		})
		duration := time.Since(start)

		if err != nil {
			fmt.Printf("similarity search failed: %v\n", err)
			continue
		}

		fmt.Printf("found %d results in %v\n\n", len(results), duration)
		for i, res := range results {
			// Fetch the raw text using the standard KV store
			text, err := c.Get(ctx, res.ID)
			if err != nil {
				text = "[Text missing or expired]"
			}
			
			fmt.Printf("--- Result %d (Score: %.4f) ---\n", i+1, res.Score)
			fmt.Printf("ID: %s\n", res.ID)
			fmt.Printf("Source: %s\n", res.Metadata["source"])
			fmt.Printf("Content: %s\n\n", text)
		}
	}
}

// loadData parses the sample JSON file and inserts it into the store.
func loadData(ctx context.Context, c *client.Client, filepath string) error {
	file, err := os.Open(filepath)
	if err != nil {
		return fmt.Errorf("could not open sample file: %w", err)
	}
	defer file.Close()

	var items []SampleItem
	if err := json.NewDecoder(file).Decode(&items); err != nil {
		return fmt.Errorf("could not decode json: %w", err)
	}

	for _, item := range items {
		// Store the raw text in the KV store
		if err := c.Set(ctx, item.ID, item.Text, 0); err != nil {
			return fmt.Errorf("failed to store text for %s: %w", item.ID, err)
		}

		// Store the vector and metadata in the Vector store
		err := c.VSet(ctx, client.VSetArgs{
			Namespace: "docs",
			ID:        item.ID,
			Vector:    item.Embedding,
			Metadata:  item.Metadata,
		})
		if err != nil {
			return fmt.Errorf("failed to store vector for %s: %w", item.ID, err)
		}
	}
	return nil
}

// getOpenAIEmbedding requests a vector embedding from the OpenAI API.
func getOpenAIEmbedding(text, apiKey string) ([]float32, error) {
	url := "https://api.openai.com/v1/embeddings"
	payload := map[string]interface{}{
		"input": text,
		"model": "text-embedding-3-small",
	}
	body, _ := json.Marshal(payload)

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var res OpenAIEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}

	if len(res.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	return res.Data[0].Embedding, nil
}

// generateRandomVector returns a normalized random float32 array.
func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	var normSq float32
	for i := 0; i < dim; i++ {
		v := rand.Float32()*2 - 1
		vec[i] = v
		normSq += v * v
	}
	norm := float32(math.Sqrt(float64(normSq)))
	for i := 0; i < dim; i++ {
		vec[i] /= norm
	}
	return vec
}
