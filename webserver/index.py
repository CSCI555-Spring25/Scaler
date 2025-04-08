#!/usr/bin/python
#authors: yeasy.github.com
#date: 2013-07-05

import sys
import BaseHTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
import socket
import fcntl
import struct
import pickle
from datetime import datetime
from collections import OrderedDict
import random
import math

class HandlerClass(SimpleHTTPRequestHandler):
    def compute_fibonacci(self):
        def fib(n):
            if n <= 1:
                return n
            return fib(n-1) + fib(n-2)
        n = random.randint(30, 35)
        return fib(n)

    def compute_matrix_multiplication(self):
        # Create two large random matrices
        size = random.randint(100, 150)
        matrix1 = [[random.random() for _ in range(size)] for _ in range(size)]
        matrix2 = [[random.random() for _ in range(size)] for _ in range(size)]
        
        # Perform matrix multiplication using pure Python
        result = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result

    def find_large_primes(self):
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        primes = []
        num = random.randint(1000000, 2000000)
        while len(primes) < 5:
            if is_prime(num):
                primes.append(num)
            num += 1
        return primes

    def compute_string_permutations(self):
        chars = 'abcdefghijklmnopqrstuvwxyz'
        n = random.randint(8, 10)
        s = ''.join(random.choices(chars, k=n))
        
        def permute(s, l, r):
            if l == r:
                return
            for i in range(l, r + 1):
                s[l], s[i] = s[i], s[l]
                permute(s, l + 1, r)
                s[l], s[i] = s[i], s[l]
        
        s_list = list(s)
        permute(s_list, 0, len(s_list) - 1)
        return s

    def compute_intensive_task(self):
        # Randomly select one of the compute-intensive tasks
        tasks = [
            self.compute_fibonacci,
            self.compute_matrix_multiplication,
            self.find_large_primes,
            self.compute_string_permutations
        ]
        return random.choice(tasks)()

    def get_ip_address(self,ifname):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', ifname[:15])
        )[20:24])

    def do_GET(self):
        """Handle GET requests by sending task information"""
        # Randomly add computational overhead to 20% of requests
        task_executed = False
        task_name = None
        result = None
        
        if random.random() < 0.5:
            task_executed = True
            tasks = {
                self.compute_fibonacci: "Fibonacci",
                self.compute_matrix_multiplication: "Matrix Multiplication",
                self.find_large_primes: "Large Primes",
                self.compute_string_permutations: "String Permutations"
            }
            chosen_task = random.choice(list(tasks.keys()))
            task_name = tasks[chosen_task]
            result = chosen_task()
            
        # Create response content
        time_now = datetime.now()
        ts = time_now.strftime('%Y-%m-%d %H:%M:%S')
        
        response_lines = [
            f"=== Request at {ts} ===",
        ]
        if task_executed:
            response_lines.extend([
                f"Compute-intensive task executed: {task_name}",
                f"Result: {result}"
            ])
        else:
            response_lines.append("No compute-intensive task executed")
        response_lines.append("=" * 40)
        
        response = "\n".join(response_lines)
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Content-length', len(response))
        self.end_headers()
        self.wfile.write(response)
        
        # Also log to file
        with open("server_log.txt", "a") as f:
            f.write(response + "\n")

    def log_message(self, format, *args):
        # Override to suppress default logging
        pass

if __name__ == '__main__':
    try:
        ServerClass  = BaseHTTPServer.HTTPServer
        Protocol     = "HTTP/1.0"
        addr = len(sys.argv) < 2 and "0.0.0.0" or sys.argv[1]
        port = len(sys.argv) < 3 and 80 or int(sys.argv[2])
        HandlerClass.protocol_version = Protocol
        httpd = ServerClass((addr, port), HandlerClass)
        sa = httpd.socket.getsockname()
        print "Serving HTTP on", sa[0], "port", sa[1], "..."
        httpd.serve_forever()
    except:
        exit()

