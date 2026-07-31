package client

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Options configures the Synapse Cache client.
type Options struct {
	Addr        string
	Password    string
	MaxConns    int
	DialTimeout time.Duration
}

// Client represents a Synapse Cache connection.
type Client struct {
	opts Options
	pool chan net.Conn
	mu   sync.Mutex
}

// New creates a new client and dials the server.
func New(opts Options) (*Client, error) {
	if opts.MaxConns <= 0 {
		opts.MaxConns = 1
	}
	if opts.DialTimeout <= 0 {
		opts.DialTimeout = 5 * time.Second
	}

	c := &Client{
		opts: opts,
		pool: make(chan net.Conn, opts.MaxConns),
	}

	// Pre-fill pool with one connection to verify connectivity
	conn, err := c.dial()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to %s: %w", opts.Addr, err)
	}
	c.pool <- conn

	return c, nil
}

// Close closes all pooled connections.
func (c *Client) Close() error {
	close(c.pool)
	for conn := range c.pool {
		conn.Close()
	}
	return nil
}

// dial opens a new authenticated TCP connection.
func (c *Client) dial() (net.Conn, error) {
	conn, err := net.DialTimeout("tcp", c.opts.Addr, c.opts.DialTimeout)
	if err != nil {
		return nil, err
	}

	if c.opts.Password != "" {
		conn.SetDeadline(time.Now().Add(c.opts.DialTimeout))
		fmt.Fprintf(conn, "AUTH %s\r\n", c.opts.Password)
		
		reader := bufio.NewReader(conn)
		line, err := reader.ReadString('\n')
		if err != nil {
			conn.Close()
			return nil, fmt.Errorf("auth failed to read response: %v", err)
		}
		if !strings.HasPrefix(line, "+OK") {
			conn.Close()
			return nil, fmt.Errorf("auth failed: %s", strings.TrimSpace(line))
		}
		conn.SetDeadline(time.Time{}) // reset deadline
	}

	return conn, nil
}

// getConn retrieves a connection from the pool or creates a new one.
// Validates connection health before returning it.
func (c *Client) getConn(ctx context.Context) (net.Conn, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case conn := <-c.pool:
			if conn != nil {
				// Perform quick health check on pooled connection
				conn.SetReadDeadline(time.Now().Add(1 * time.Millisecond))
				var one [1]byte
				_, err := conn.Read(one[:])
				conn.SetReadDeadline(time.Time{})
				if err != nil {
					if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
						// Socket is healthy (timeout expected on empty buffer read)
						return conn, nil
					}
					// Connection closed or broken by server, discard
					conn.Close()
					continue
				}
				// Unexpected data on idle socket, discard
				conn.Close()
				continue
			}
		default:
			return c.dial()
		}
	}
}

// returnConn returns a connection to the pool.
func (c *Client) returnConn(conn net.Conn) {
	select {
	case c.pool <- conn:
	default:
		conn.Close() // pool full, discard
	}
}

// RespValue represents a parsed RESP response.
type RespValue struct {
	IsNil bool
	Str   string
	Array []RespValue
}

// RawExec executes a raw command.
func (c *Client) RawExec(ctx context.Context, cmd string) (RespValue, error) {
	conn, err := c.getConn(ctx)
	if err != nil {
		return RespValue{}, err
	}

	deadline, ok := ctx.Deadline()
	if ok {
		conn.SetDeadline(deadline)
	} else {
		conn.SetDeadline(time.Now().Add(5 * time.Second))
	}

	_, err = fmt.Fprint(conn, cmd+"\r\n")
	if err != nil {
		conn.Close()
		return RespValue{}, fmt.Errorf("write error: %w", err)
	}

	reader := bufio.NewReader(conn)
	resp, err := readResponse(reader)
	if err != nil {
		conn.Close()
		return RespValue{}, err
	}

	c.returnConn(conn)
	return resp, nil
}

// readResponse decodes a RESP payload.
func readResponse(reader *bufio.Reader) (RespValue, error) {
	line, err := reader.ReadString('\n')
	if err != nil {
		return RespValue{}, err
	}
	line = strings.TrimRight(line, "\r\n")

	if len(line) == 0 {
		return RespValue{}, fmt.Errorf("empty response")
	}

	prefix := line[0]
	body := line[1:]

	switch prefix {
	case '+':
		return RespValue{Str: body}, nil
	case '-':
		return RespValue{}, fmt.Errorf("server error: %s", body)
	case ':':
		return RespValue{Str: body}, nil
	case '$':
		length, err := strconv.Atoi(body)
		if err != nil {
			return RespValue{}, fmt.Errorf("invalid bulk string length: %s", body)
		}
		if length == -1 {
			return RespValue{IsNil: true}, ErrNil
		}
		// Read the data + \r\n
		data := make([]byte, length+2)
		_, err = io.ReadFull(reader, data)
		if err != nil {
			return RespValue{}, err
		}
		return RespValue{Str: string(data[:length])}, nil
	case '*':
		// Array — read recursively
		count, err := strconv.Atoi(body)
		if err != nil {
			return RespValue{}, fmt.Errorf("invalid array count: %s", body)
		}
		if count <= 0 {
			return RespValue{Array: []RespValue{}}, nil
		}
		var items []RespValue
		for i := 0; i < count; i++ {
			item, err := readResponse(reader)
			if err != nil && err != ErrNil {
				return RespValue{}, err
			}
			items = append(items, item)
		}
		return RespValue{Array: items}, nil
	default:
		return RespValue{Str: line}, nil
	}
}

// --- Error types ---

// ErrNil indicates a cache miss.
var ErrNil = fmt.Errorf("nil")

// --- Key-Value Operations ---

// Set stores a key-value pair with optional TTL.
func (c *Client) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	cmd := fmt.Sprintf("SET %s %s", key, value)
	if ttl > 0 {
		cmd += fmt.Sprintf(" EX %d", int(ttl.Seconds()))
	}
	_, err := c.RawExec(ctx, cmd)
	return err
}

// Get retrieves a value by key.
func (c *Client) Get(ctx context.Context, key string) (string, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("GET %s", key))
	if err != nil {
		return "", err
	}
	return resp.Str, nil
}

// Del deletes keys and returns the count deleted.
func (c *Client) Del(ctx context.Context, keys ...string) (int, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("DEL %s", strings.Join(keys, " ")))
	if err != nil {
		return 0, err
	}
	n, _ := strconv.Atoi(resp.Str)
	return n, nil
}

// Expire sets a key's TTL.
func (c *Client) Expire(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("EXPIRE %s %d", key, int(ttl.Seconds())))
	if err != nil {
		return false, err
	}
	return resp.Str == "1", nil
}

// TTL returns the remaining time to live of a key.
func (c *Client) TTL(ctx context.Context, key string) (int, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("TTL %s", key))
	if err != nil {
		return 0, err
	}
	n, _ := strconv.Atoi(resp.Str)
	return n, nil
}

// Info returns the server information.
func (c *Client) Info(ctx context.Context) (string, error) {
	resp, err := c.RawExec(ctx, "INFO")
	if err != nil {
		return "", err
	}
	return resp.Str, nil
}

// --- Vector Operations ---

// VSetArgs holds arguments for the VSet command.
type VSetArgs struct {
	Namespace string
	ID        string
	Vector    []float32
	Metadata  map[string]string
}

// VSet inserts a vector.
func (c *Client) VSet(ctx context.Context, args VSetArgs) error {
	parts := []string{"VSET", args.Namespace, args.ID, strconv.Itoa(len(args.Vector))}
	for _, f := range args.Vector {
		parts = append(parts, fmt.Sprintf("%f", f))
	}
	if len(args.Metadata) > 0 {
		parts = append(parts, "META")
		for k, v := range args.Metadata {
			parts = append(parts, k, v)
		}
	}
	_, err := c.RawExec(ctx, strings.Join(parts, " "))
	return err
}

// VSimilarityArgs holds arguments for the VSimilarity command.
type VSimilarityArgs struct {
	Namespace string
	Vector    []float32
	TopK      int
}

// SimilarityResult represents a single similarity search result.
type SimilarityResult struct {
	ID       string
	Score    float32
	Metadata map[string]string
}

// VGet retrieves a vector. Returns ErrNil if not found.
func (c *Client) VGet(ctx context.Context, namespace, id string) ([]float32, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("VGET %s %s", namespace, id))
	if err != nil {
		return nil, err
	}

	if len(resp.Array) == 0 && resp.IsNil {
		return nil, ErrNil
	}

	vec := make([]float32, len(resp.Array))
	for i, item := range resp.Array {
		f, _ := strconv.ParseFloat(item.Str, 32)
		vec[i] = float32(f)
	}
	return vec, nil
}

// VDel deletes a vector.
func (c *Client) VDel(ctx context.Context, namespace, id string) (bool, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("VDEL %s %s", namespace, id))
	if err != nil {
		return false, err
	}
	return resp.Str == "1", nil
}

// VSimilarity executes a Top-K cosine similarity search.
func (c *Client) VSimilarity(ctx context.Context, args VSimilarityArgs) ([]SimilarityResult, error) {
	parts := []string{"VSIMILARITY", args.Namespace, strconv.Itoa(len(args.Vector))}
	for _, f := range args.Vector {
		parts = append(parts, fmt.Sprintf("%f", f))
	}
	parts = append(parts, "TOP", strconv.Itoa(args.TopK))

	resp, err := c.RawExec(ctx, strings.Join(parts, " "))
	if err != nil {
		return nil, err
	}

	if len(resp.Array) == 0 {
		return nil, nil
	}

	// Result is an array where each match is 3 items: [ID, Score, [MetaKey, MetaVal, ...]]
	var results []SimilarityResult
	for i := 0; i+2 < len(resp.Array); i += 3 {
		id := resp.Array[i].Str
		score, _ := strconv.ParseFloat(resp.Array[i+1].Str, 32)

		result := SimilarityResult{
			ID:    id,
			Score: float32(score),
		}

		metaArray := resp.Array[i+2].Array
		if len(metaArray) > 0 {
			result.Metadata = make(map[string]string)
			for j := 0; j+1 < len(metaArray); j += 2 {
				result.Metadata[metaArray[j].Str] = metaArray[j+1].Str
			}
		}
		results = append(results, result)
	}

	return results, nil
}

// VCount returns the number of vectors in a namespace.
func (c *Client) VCount(ctx context.Context, namespace string) (int, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("VCOUNT %s", namespace))
	if err != nil {
		return 0, err
	}
	n, _ := strconv.Atoi(resp.Str)
	return n, nil
}

// VIndexCreateArgs holds arguments for the VINDEX CREATE command.
type VIndexCreateArgs struct {
	Namespace      string
	M              int
	EfConstruction int
	EfSearch       int
}

// VIndexCreate creates a new HNSW index for the given namespace.
func (c *Client) VIndexCreate(ctx context.Context, args VIndexCreateArgs) error {
	parts := []string{"VINDEX", "CREATE", args.Namespace}
	if args.M > 0 {
		parts = append(parts, "M", strconv.Itoa(args.M))
	}
	if args.EfConstruction > 0 {
		parts = append(parts, "EF_CONSTRUCTION", strconv.Itoa(args.EfConstruction))
	}
	if args.EfSearch > 0 {
		parts = append(parts, "EF_SEARCH", strconv.Itoa(args.EfSearch))
	}
	_, err := c.RawExec(ctx, strings.Join(parts, " "))
	return err
}

// VIndexDrop drops an existing HNSW index.
func (c *Client) VIndexDrop(ctx context.Context, namespace string) error {
	_, err := c.RawExec(ctx, fmt.Sprintf("VINDEX DROP %s", namespace))
	return err
}

// VIndexInfo returns information about an HNSW index.
func (c *Client) VIndexInfo(ctx context.Context, namespace string) (string, error) {
	resp, err := c.RawExec(ctx, fmt.Sprintf("VINDEX INFO %s", namespace))
	if err != nil {
		return "", err
	}
	return resp.Str, nil
}

// VIndexSetEf sets the ef_search parameter for an HNSW index.
func (c *Client) VIndexSetEf(ctx context.Context, namespace string, ef int) error {
	_, err := c.RawExec(ctx, fmt.Sprintf("VINDEX SET_EF %s %d", namespace, ef))
	return err
}

// Ping sends a PING command.
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.RawExec(ctx, "PING")
	return err
}
