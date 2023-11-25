import http.server
import socketserver
import urllib.parse as urlparse

PORT = 8000

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Check if the root of the server is requested
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            message = "Welcome to the Twitch Recommender System"
            self.wfile.write(bytes(f"<html><head><title>Twitch Recommender System</title></head><body><h1>{message}</h1></body></html>", "utf-8"))
        elif 'code' in urlparse.parse_qs(urlparse.urlparse(self.path).query):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            authorization_code = urlparse.parse_qs(urlparse.urlparse(self.path).query)['code'][0]
            self.wfile.write(bytes(f"<html><head><title>Authorization Received</title></head><body><h1>Twitch Recommender System</h1><p>Your authorization code: {authorization_code}</p></body></html>", "utf-8"))
        else:
            # Handle any other GET requests
            self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes("<html><head><title>Not Found</title></head><body><p>Page not found</p></body></html>", "utf-8"))

handler_object = MyHttpRequestHandler

with socketserver.TCPServer(("", PORT), handler_object) as httpd:
    print(f"Server started at localhost:{PORT}")
    httpd.serve_forever()
