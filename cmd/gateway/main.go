package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/pkg/client"
)

// SearchRequest represents the JSON payload from the frontend
type SearchRequest struct {
	Query     string `json:"query"`
	Namespace string `json:"namespace"`
	TopK      int    `json:"top_k"`
}

// SearchResponse represents the JSON payload back to the frontend
type SearchResponse struct {
	Latency float64       `json:"latency"`
	Results []SearchResult `json:"results"`
}

type SearchResult struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"`
	Metadata map[string]string `json:"metadata"`
	Text     string            `json:"text"`
}

type OpenAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func main() {
	httpPort := flag.String("port", "8080", "HTTP Server Port")
	dbAddr := flag.String("db-addr", "localhost:6379", "Synapse Cache TCP Address")
	dbPass := flag.String("db-pass", "", "Synapse Cache Password")
	openAIKey := flag.String("openai-key", "", "OpenAI API Key (required for real embeddings)")
	staticDir := flag.String("static", "./frontend", "Path to frontend static files")
	flag.Parse()

	log.Println("Starting Synapse HTTP Gateway...")

	// Connect to the TCP Database
	dbClient, err := client.New(client.Options{
		Addr:        *dbAddr,
		Password:    *dbPass,
		MaxConns:    10,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Synapse Database at %s: %v", *dbAddr, err)
	}
	defer dbClient.Close()
	log.Printf("Connected to Synapse TCP Database at %s", *dbAddr)

	if *openAIKey == "" {
		log.Println("WARNING: No --openai-key provided. Search will use zero-vectors and fail semantic matching.")
	}

	// Handlers
	http.HandleFunc("/api/search", func(w http.ResponseWriter, r *http.Request) {
		handleSearch(w, r, dbClient, *openAIKey)
	})

	http.HandleFunc("/api/stats", func(w http.ResponseWriter, r *http.Request) {
		handleStats(w, r, dbClient)
	})

	// Serve Static Files
	fs := http.FileServer(http.Dir(*staticDir))
	http.Handle("/", fs)

	log.Printf("HTTP Gateway serving frontend at http://localhost:%s", *httpPort)
	if err := http.ListenAndServe(":"+*httpPort, nil); err != nil {
		log.Fatalf("HTTP server failed: %v", err)
	}
}

func handleSearch(w http.ResponseWriter, r *http.Request, db *client.Client, openAIKey string) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Namespace == "" {
		req.Namespace = "docs"
	}
	if req.TopK <= 0 {
		req.TopK = 3
	}

	start := time.Now()
	var queryVec []float32
	var err error

	if openAIKey != "" {
		queryVec, err = getOpenAIEmbedding(req.Query, openAIKey)
		if err != nil {
			http.Error(w, fmt.Sprintf("OpenAI API error: %v", err), http.StatusInternalServerError)
			return
		}
	} else {
		// Fallback dummy vector if no API key is provided
		queryVec = make([]float32, 1536)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// 1. VSIMILARITY TCP Call
	simResults, err := db.VSimilarity(ctx, client.VSimilarityArgs{
		Namespace: req.Namespace,
		Vector:    queryVec,
		TopK:      req.TopK,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Database query failed: %v", err), http.StatusInternalServerError)
		return
	}

	// 2. Fetch TEXT bodies via GET TCP Call
	var finalResults []SearchResult
	for _, res := range simResults {
		text, err := db.Get(ctx, res.ID)
		if err != nil {
			text = "" // Key expired or missing
		}
		finalResults = append(finalResults, SearchResult{
			ID:       res.ID,
			Score:    res.Score,
			Metadata: res.Metadata,
			Text:     text,
		})
	}

	duration := time.Since(start).Seconds() * 1000.0 // ms

	resp := SearchResponse{
		Latency: duration,
		Results: finalResults,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleStats(w http.ResponseWriter, r *http.Request, db *client.Client) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	count, err := db.VCount(ctx, "docs")
	if err != nil {
		http.Error(w, fmt.Sprintf("VCOUNT failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int{"vectors": count})
}

// getOpenAIEmbedding calls the real OpenAI API to embed a user query
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

	hc := &http.Client{Timeout: 5 * time.Second}
	resp, err := hc.Do(req)
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
