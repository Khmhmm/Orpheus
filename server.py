from ml import model
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import json

class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.predicting_model = model.load()
        super(RequestHandler, self).__init__(*args, **kwargs)

    def do_POST(self):
        content_len = int(self.headers.get('content-length'))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body)

        predicted_class = self.predicting_model.predict([data])
        #cast from np.array to python.array
        cl = [float(predicted_class[0])]

        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(
            cl
        ).encode())
        return

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8060), RequestHandler)
    print('Starting server at http://localhost:8060')
    server.serve_forever()
