package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/pkg/client"
)

func main() {
	addr := flag.String("addr", "localhost:6379", "Server address")
	auth := flag.String("auth", "", "Password for authentication")
	flag.Parse()

	c, err := client.New(client.Options{
		Addr:     *addr,
		Password: *auth,
	})
	if err != nil {
		log.Fatalf("failed to connect: %v", err)
	}
	defer c.Close()

	args := flag.Args()

	if len(args) > 0 {
		// Single command mode
		cmd := strings.Join(args, " ")
		runCommand(c, cmd)
		return
	}

	// Interactive REPL
	fmt.Printf("Connected to Synapse Cache at %s\n", *addr)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("synapse> ")
		if !scanner.Scan() {
			break
		}
		cmd := strings.TrimSpace(scanner.Text())
		if cmd == "" {
			continue
		}
		if strings.ToUpper(cmd) == "EXIT" || strings.ToUpper(cmd) == "QUIT" {
			break
		}
		runCommand(c, cmd)
	}
}

func runCommand(c *client.Client, cmd string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.RawExec(ctx, cmd)
	if err != nil {
		fmt.Printf("(error) %v\n", err)
		return
	}
	if resp.IsNil {
		fmt.Println("(nil)")
		return
	}
	fmt.Println(resp.Str)
}
