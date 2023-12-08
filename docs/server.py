'''Basic web server serving static files from the docs directory.'''

import os
import sys
import time
import http.server
import socketserver

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

os.chdir('docs')

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()

