package server

import (
	"strconv"
	"strings"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/protocol"
)

func (srv *Server) handleSet(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) < 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'SET' command"}
	}

	key := cmd.Args[0]
	value := cmd.Args[1]

	// Check for optional EX <seconds>
	if len(cmd.Args) >= 4 && strings.ToUpper(cmd.Args[2]) == "EX" {
		seconds, err := strconv.Atoi(cmd.Args[3])
		if err != nil || seconds <= 0 {
			return &protocol.ErrorResponse{Message: "invalid expire time in 'SET' command"}
		}
		srv.store.SetWithTTL(key, value, time.Duration(seconds)*time.Second)
	} else {
		srv.store.Set(key, value)
	}

	return protocol.RespOK
}

func (srv *Server) handleGet(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) != 1 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'GET' command"}
	}

	val, ok := srv.store.Get(cmd.Args[0])
	if !ok {
		return protocol.RespNil
	}
	return &protocol.BulkString{Value: val}
}

func (srv *Server) handleDel(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) == 0 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'DEL' command"}
	}
	count := srv.store.Del(cmd.Args...)
	return &protocol.IntegerResponse{Value: count}
}

func (srv *Server) handleExpire(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) != 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'EXPIRE' command"}
	}
	seconds, err := strconv.Atoi(cmd.Args[1])
	if err != nil || seconds <= 0 {
		return &protocol.ErrorResponse{Message: "invalid expire time"}
	}
	ok := srv.store.Expire(cmd.Args[0], time.Duration(seconds)*time.Second)
	if ok {
		return &protocol.IntegerResponse{Value: 1}
	}
	return &protocol.IntegerResponse{Value: 0}
}

func (srv *Server) handleTTL(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) != 1 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'TTL' command"}
	}
	ttl := srv.store.TTL(cmd.Args[0])
	return &protocol.IntegerResponse{Value: ttl}
}
