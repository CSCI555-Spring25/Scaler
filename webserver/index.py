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
    def log_message(self, format, *args):
        # Randomly add computational overhead to 20% of requests
        if random.random() < 0.5:
            self.compute_intensive_task()
            
        if len(args) < 3 or "200" not in args[1]:
            return
        try:
            request = pickle.load(open("pickle_data.txt","r"))
        except:
            request=OrderedDict()
        time_now = datetime.now()
        ts = time_now.strftime('%Y-%m-%d %H:%M:%S')
        server = self.get_ip_address('eth0')
        host=self.address_string()
        addr_pair = (host,server)
        if addr_pair not in request:
            request[addr_pair]=[1,ts]
        else:
            num = request[addr_pair][0]+1
            del request[addr_pair]
            request[addr_pair]=[num,ts]
        file=open("index.html", "w")
        file.write("<!DOCTYPE html> <html> <body><center><h1><font color=\"blue\" face=\"Georgia, Arial\" size=8><em>Real</em></font> Visit Results</h1></center>");
        for pair in request:
            if pair[0] == host:
                guest = "LOCAL: "+pair[0]
            else:
                guest = pair[0]
            if (time_now-datetime.strptime(request[pair][1],'%Y-%m-%d %H:%M:%S')).seconds < 3:
                file.write("<p style=\"font-size:150%\" >#"+ str(request[pair][1]) +": <font color=\"red\">"+str(request[pair][0])+ "</font> requests " + "from &lt<font color=\"blue\">"+guest+"</font>&gt to WebServer &lt<font color=\"blue\">"+pair[1]+"</font>&gt</p>")
            else:
                file.write("<p style=\"font-size:150%\" >#"+ str(request[pair][1]) +": <font color=\"maroon\">"+str(request[pair][0])+ "</font> requests " + "from &lt<font color=\"navy\">"+guest+"</font>&gt to WebServer &lt<font color=\"navy\">"+pair[1]+"</font>&gt</p>")
        file.write("</body> </html>");
        file.close()
        pickle.dump(request,open("pickle_data.txt","w"))

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

