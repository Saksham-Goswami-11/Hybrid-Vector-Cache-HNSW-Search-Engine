import socket
import ssl
import threading
import os
from typing import List, Dict, Any, Optional, Union

class SynapseError(Exception):
    pass

class SynapseClient:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.environ.get("NEARBY_HOST", "localhost")
        self.port = port if port is not None else int(os.environ.get("NEARBY_PORT", 6380))
        self.password = os.environ.get("NEARBY_PASSWORD")
        self.lock = threading.Lock()

    def _readline(self, sock: socket.socket) -> bytes:
        line = bytearray()
        while True:
            char = sock.recv(1)
            if not char:
                raise ConnectionError("Connection closed by server")
            line.extend(char)
            if len(line) >= 2 and line[-2:] == b'\r\n':
                return bytes(line[:-2])

    def _read_response(self, sock: socket.socket) -> Any:
        line = self._readline(sock)
        if not line:
            raise SynapseError("Received empty response line")
        
        prefix = chr(line[0])
        body = line[1:].decode('utf-8')

        if prefix == '+':
            return body
        elif prefix == '-':
            raise SynapseError(f"Server error: {body}")
        elif prefix == ':':
            return int(body)
        elif prefix == '$':
            length = int(body)
            if length == -1:
                return None
            data = sock.recv(length)
            # read the trailing \r\n
            crlf = sock.recv(2)
            return data.decode('utf-8')
        elif prefix == '*':
            count = int(body)
            if count == -1:
                return None
            items = []
            for _ in range(count):
                items.append(self._read_response(sock))
            return items
        else:
            raise SynapseError(f"Unknown RESP protocol prefix: {prefix}")

    def _send_command(self, cmd_string: str) -> Any:
        with self.lock:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw_socket:
                s = raw_socket
                if os.environ.get("NEARBY_TLS", "false").lower() == "true":
                    context = ssl.create_default_context()
                    if os.environ.get("NEARBY_INSECURE_SKIP_VERIFY", "false").lower() == "true":
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                    s = context.wrap_socket(raw_socket, server_hostname=self.host)
                
                s.connect((self.host, self.port))
                
                # Perform AUTH handshake if password is set
                if self.password:
                    auth_cmd = f"AUTH {self.password}\r\n".encode('utf-8')
                    s.sendall(auth_cmd)
                    auth_res = self._read_response(s)
                    if auth_res != "OK":
                        raise SynapseError(f"Authentication failed: {auth_res}")

                s.sendall(cmd_string.encode('utf-8') + b'\r\n')
                return self._read_response(s)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        quoted_key = self._quote(key)
        quoted_val = self._quote(value)
        cmd = f"SET {quoted_key} {quoted_val}"
        if ex is not None:
            cmd += f" EX {ex}"
        res = self._send_command(cmd)
        return res == "OK"

    def get(self, key: str) -> Optional[str]:
        quoted_key = self._quote(key)
        try:
            return self._send_command(f"GET {quoted_key}")
        except SynapseError as e:
            if "nil" in str(e).lower():
                return None
            raise

    def vset(self, namespace: str, key: str, vector: List[float], metadata: Optional[Dict[str, str]] = None) -> bool:
        dim = len(vector)
        vector_str = " ".join(f"{f:.6f}" for f in vector)
        quoted_ns = self._quote(namespace)
        quoted_key = self._quote(key)
        cmd = f"VSET {quoted_ns} {quoted_key} {dim} {vector_str}"
        if metadata:
            meta_parts = []
            for k, v in metadata.items():
                meta_parts.append(self._quote(k))
                meta_parts.append(self._quote(v))
            cmd += " META " + " ".join(meta_parts)
        res = self._send_command(cmd)
        return res == "OK"

    def vsimilarity(self, namespace: str, vector: List[float], k: int) -> List[Dict[str, Any]]:
        dim = len(vector)
        vector_str = " ".join(f"{f:.6f}" for f in vector)
        quoted_ns = self._quote(namespace)
        cmd = f"VSIMILARITY {quoted_ns} {dim} {vector_str} TOP {k}"
        res = self._send_command(cmd)
        if not res:
            return []
        
        results = []
        i = 0
        while i + 2 < len(res):
            doc_id = res[i]
            score = float(res[i+1])
            meta_list = res[i+2]
            
            metadata = {}
            if isinstance(meta_list, list):
                for j in range(0, len(meta_list), 2):
                    if j+1 < len(meta_list):
                        metadata[meta_list[j]] = meta_list[j+1]
            
            results.append({
                "id": doc_id,
                "score": score,
                "metadata": metadata
            })
            i += 3
            
        return results

    def _quote(self, s: str) -> str:
        # Always escape properly when quoting
        if ' ' in s or '"' in s or "'" in s or '\n' in s or '\r' in s or '\t' in s or '\\' in s or s == "":
            escaped = s.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"{escaped}"'
        return s
